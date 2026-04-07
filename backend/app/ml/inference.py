from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from app.config import Settings
from app.feature_schema import OPTIONAL_LAB_KEYS, REQUIRED_INPUT_KEYS
from app.ml.training import MODEL_ARTIFACT_NAME, MODEL_FEATURE_COLUMNS, train_and_save_model
from app.schemas import NormalizedFeatures, PredictionInput, PredictionResponse, RecommendationItem


MMOL_TO_MGDL = 18.0182


@dataclass
class ModelService:
    settings: Settings
    model: object | None = None
    models: dict[str, object] | None = None
    metadata: dict | None = None

    def load(self) -> None:
        artifact_path = self.settings.model_dir / MODEL_ARTIFACT_NAME
        if not artifact_path.exists():
            result = train_and_save_model(
                model_dir=self.settings.model_dir,
                data_dir=self.settings.data_dir,
                dataset_url=self.settings.model_data_url,
                target_column=self.settings.model_target_column,
            )
            self.metadata = result.metadata
        artifact = joblib.load(artifact_path)
        if "models" not in artifact or not artifact.get("metadata", {}).get("all_model_metrics"):
            result = train_and_save_model(
                model_dir=self.settings.model_dir,
                data_dir=self.settings.data_dir,
                dataset_url=self.settings.model_data_url,
                target_column=self.settings.model_target_column,
            )
            self.metadata = result.metadata
            artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.models = artifact.get("models") or {
            artifact.get("metadata", {}).get("model_name", "default"): artifact["model"]
        }
        self.metadata = artifact["metadata"]

    @property
    def ready(self) -> bool:
        return self.models is not None and self.metadata is not None

    def _derive_glucose(self, raw: dict) -> tuple[float | None, float | None]:
        glucose_mg_dl = raw.get("fasting_glucose_mg_dl")
        glucose_mmol_l = raw.get("fasting_glucose_mmol_l")
        if glucose_mg_dl is None and glucose_mmol_l is not None:
            glucose_mg_dl = round(glucose_mmol_l * MMOL_TO_MGDL, 2)
        if glucose_mg_dl is None and raw.get("hba1c") is not None:
            glucose_mg_dl = round((28.7 * raw["hba1c"]) - 46.7, 2)
        if glucose_mmol_l is None and glucose_mg_dl is not None:
            glucose_mmol_l = round(glucose_mg_dl / MMOL_TO_MGDL, 2)
        return glucose_mg_dl, glucose_mmol_l

    def _derive_bmi(self, raw: dict) -> float | None:
        if raw.get("bmi") is not None:
            return raw["bmi"]
        height_cm = raw.get("height_cm")
        weight_kg = raw.get("weight_kg")
        if height_cm and weight_kg:
            height_m = height_cm / 100
            if height_m > 0:
                return round(weight_kg / (height_m**2), 2)
        return None

    def _derive_insulin(self, raw: dict, glucose_mmol_l: float | None) -> float | None:
        insulin = raw.get("insulin")
        if insulin is not None:
            return insulin
        homa_ir = raw.get("homa_ir")
        if homa_ir is not None and glucose_mmol_l and glucose_mmol_l > 0:
            return round((homa_ir * 22.5) / glucose_mmol_l, 2)
        return None

    def normalize_input(
        self, payload: PredictionInput
    ) -> tuple[dict[str, float | int], list[str], dict]:
        raw = payload.model_dump()
        glucose_mg_dl, glucose_mmol_l = self._derive_glucose(raw)
        bmi = self._derive_bmi(raw)
        insulin = self._derive_insulin(raw, glucose_mmol_l)

        required_normalized = {
            "age": raw.get("age"),
            "sex": raw.get("sex"),
            "height_cm": raw.get("height_cm"),
            "weight_kg": raw.get("weight_kg"),
            "systolic_bp": raw.get("systolic_bp"),
            "diastolic_bp": raw.get("diastolic_bp"),
            "physical_activity_days_per_week": raw.get("physical_activity_days_per_week"),
            "smoking": raw.get("smoking"),
            "family_history_diabetes": raw.get("family_history_diabetes"),
            "hypertension": raw.get("hypertension"),
            "cardiovascular_disease": raw.get("cardiovascular_disease"),
            "stroke_history": raw.get("stroke_history"),
        }
        missing = [key for key in REQUIRED_INPUT_KEYS if required_normalized.get(key) is None]

        derived_pedigree = 0.25
        if raw.get("family_history_diabetes"):
            derived_pedigree += 0.45
        if raw.get("hypertension"):
            derived_pedigree += 0.12
        if raw.get("cardiovascular_disease"):
            derived_pedigree += 0.10
        if raw.get("smoking"):
            derived_pedigree += 0.08
        if raw.get("stroke_history"):
            derived_pedigree += 0.06

        physical_days = raw.get("physical_activity_days_per_week") or 0
        derived_skin_thickness = round(max(12.0, min(45.0, (bmi or 24) * 0.65 - physical_days * 0.4)), 2)
        derived_diastolic = raw.get("diastolic_bp") or (86 if raw.get("hypertension") else 74)
        derived_insulin = insulin if insulin is not None else (125 if raw.get("hypertension") else 85)
        derived_glucose = glucose_mg_dl if glucose_mg_dl is not None else 108.0
        derived_pregnancies = raw.get("pregnancy_count") or 0

        derived_model_features = {
            "Pregnancies": derived_pregnancies,
            "Glucose": round(derived_glucose, 2),
            "BloodPressure": int(derived_diastolic),
            "SkinThickness": derived_skin_thickness,
            "Insulin": round(derived_insulin, 2),
            "BMI": round(bmi or 26.5, 2),
            "DiabetesPedigreeFunction": round(min(2.5, derived_pedigree), 3),
            "Age": int(raw.get("age") or 45),
        }

        enriched = {
            **raw,
            "bmi": bmi,
            "fasting_glucose_mg_dl": glucose_mg_dl,
            "fasting_glucose_mmol_l": glucose_mmol_l,
            "insulin": insulin,
            "derived_model_features": derived_model_features,
        }
        return derived_model_features, missing, enriched

    def _frame_for_model(self, derived_features: dict[str, float | int]) -> pd.DataFrame:
        return pd.DataFrame([derived_features], columns=MODEL_FEATURE_COLUMNS)

    def _build_explanation(self, normalized: dict) -> tuple[str, list[str]]:
        factors: list[str] = []
        glucose = normalized.get("fasting_glucose_mg_dl")
        bmi = normalized.get("bmi")
        if glucose and glucose >= 126:
            factors.append("fasting glucose is in an elevated range")
        elif glucose and glucose >= 100:
            factors.append("fasting glucose is mildly elevated")
        if normalized.get("hba1c") and normalized["hba1c"] >= 6.5:
            factors.append("HbA1c is elevated")
        if bmi and bmi >= 30:
            factors.append("BMI is in an obesity range")
        elif bmi and bmi >= 25:
            factors.append("BMI is above the healthy range")
        if normalized.get("hypertension"):
            factors.append("hypertension history is present")
        if normalized.get("family_history_diabetes"):
            factors.append("family history of diabetes is present")
        if normalized.get("cardiovascular_disease"):
            factors.append("cardiovascular disease history is present")
        if not factors:
            factors.append("current inputs do not show strong high-risk markers")

        explanation = (
            "This estimate reflects the entered vitals, history, and any available lab values. "
            f"The strongest contributors in this assessment are: {', '.join(factors[:4])}."
        )
        return explanation, factors

    def _build_recommendations(self, normalized: dict, risk_band: str) -> list[RecommendationItem]:
        recommendations: list[RecommendationItem] = []
        bmi = normalized.get("bmi")
        glucose = normalized.get("fasting_glucose_mg_dl")
        hba1c = normalized.get("hba1c")
        homa_ir = normalized.get("homa_ir")
        activity_days = normalized.get("physical_activity_days_per_week") or 0
        cholesterol = normalized.get("cholesterol_total")

        if normalized.get("smoking"):
            recommendations.append(
                RecommendationItem(
                    title="Smoking cessation should be prioritized",
                    rationale="Current smoking status contributes to cardiometabolic risk and adds avoidable vascular burden.",
                    priority="high",
                    category="lifestyle",
                )
            )
        if bmi is not None and bmi >= 25:
            recommendations.append(
                RecommendationItem(
                    title="Weight reduction may improve metabolic risk",
                    rationale=f"Calculated BMI is {bmi:.1f}, which is above the healthy range and may contribute to insulin resistance.",
                    priority="high" if bmi >= 30 else "medium",
                    category="lifestyle",
                )
            )
        if activity_days < 3:
            recommendations.append(
                RecommendationItem(
                    title="Increase weekly physical activity",
                    rationale="Current activity frequency is low; regular aerobic and resistance exercise may improve glucose control and weight management.",
                    priority="medium",
                    category="lifestyle",
                )
            )
        if normalized.get("family_history_diabetes"):
            recommendations.append(
                RecommendationItem(
                    title="Monitor with more frequent blood tests",
                    rationale="A positive family history increases background risk, so periodic fasting glucose and HbA1c follow-up may be reasonable.",
                    priority="medium",
                    category="monitoring",
                )
            )
        if glucose is not None and glucose >= 100:
            recommendations.append(
                RecommendationItem(
                    title="Repeat fasting glucose monitoring",
                    rationale="Fasting glucose is above the normal range, so repeat testing can help confirm whether this is persistent.",
                    priority="high" if glucose >= 126 else "medium",
                    category="laboratory",
                )
            )
        if hba1c is not None and hba1c >= 5.7:
            recommendations.append(
                RecommendationItem(
                    title="Follow HbA1c over time",
                    rationale="HbA1c is elevated relative to the normal range, making longitudinal monitoring useful for trend assessment.",
                    priority="high" if hba1c >= 6.5 else "medium",
                    category="laboratory",
                )
            )
        if homa_ir is not None and homa_ir >= 2.5:
            recommendations.append(
                RecommendationItem(
                    title="Monitor insulin resistance markers",
                    rationale="HOMA-IR suggests reduced insulin sensitivity and supports closer metabolic follow-up.",
                    priority="medium",
                    category="laboratory",
                )
            )
        if normalized.get("hypertension") or normalized.get("cardiovascular_disease") or normalized.get("stroke_history"):
            recommendations.append(
                RecommendationItem(
                    title="Maintain regular cardiovascular follow-up",
                    rationale="Existing cardiovascular comorbidity strengthens the case for coordinated blood pressure and metabolic monitoring.",
                    priority="high",
                    category="medical_follow_up",
                )
            )
        if cholesterol is not None and cholesterol >= 200:
            recommendations.append(
                RecommendationItem(
                    title="Review lipid profile and diet quality",
                    rationale="Total cholesterol is elevated and may justify dietary review and repeat lipid monitoring.",
                    priority="medium",
                    category="laboratory",
                )
            )
        if risk_band == "high":
            recommendations.append(
                RecommendationItem(
                    title="Seek formal clinical review",
                    rationale="The estimated risk is high enough that professional interpretation and confirmatory testing are appropriate.",
                    priority="high",
                    category="medical_follow_up",
                )
            )

        if not recommendations:
            recommendations.append(
                RecommendationItem(
                    title="Continue periodic preventive monitoring",
                    rationale="Current inputs do not show a strong abnormal signal, but routine preventive follow-up remains appropriate.",
                    priority="low",
                    category="monitoring",
                )
            )

        unique: list[RecommendationItem] = []
        seen_titles: set[str] = set()
        for item in recommendations:
            if item.title in seen_titles:
                continue
            seen_titles.add(item.title)
            unique.append(item)
        return unique

    def predict(self, payload: PredictionInput, model_name: str | None = None) -> PredictionResponse:
        if not self.ready:
            self.load()

        derived_features, missing, enriched = self.normalize_input(payload)
        explanation, factors = self._build_explanation(enriched)
        available_models = self.models or {}
        selected_model_name = model_name or self.metadata.get("model_name") or next(iter(available_models))
        selected_model = available_models.get(selected_model_name) or self.model
        selected_metrics = (self.metadata or {}).get("all_model_metrics", {}).get(
            selected_model_name,
            (self.metadata or {}).get("metrics", {}),
        )

        if missing:
            return PredictionResponse(
                model_name=selected_model_name,
                risk_probability=0.0,
                risk_percent=0.0,
                risk_band="incomplete",
                prediction_confidence=0.0,
                explanation="Complete the required intake fields before the risk estimate can be calculated.",
                key_factors=factors,
                normalized_input=NormalizedFeatures(**enriched),
                missing_required_fields=missing,
                model_metrics=selected_metrics,
                recommendations=self._build_recommendations(enriched, "incomplete"),
            )

        frame = self._frame_for_model(derived_features)
        probability = float(selected_model.predict_proba(frame)[0][1])
        model_confidence = max(probability, 1 - probability)
        optional_labs_provided = sum(1 for key in OPTIONAL_LAB_KEYS if enriched.get(key) is not None)
        data_richness = optional_labs_provided / max(1, len(OPTIONAL_LAB_KEYS))
        prediction_confidence = ((model_confidence * 0.7) + (data_richness * 0.3)) * 100

        if probability >= 0.66:
            risk_band = "high"
        elif probability >= 0.33:
            risk_band = "moderate"
        else:
            risk_band = "low"

        recommendations = self._build_recommendations(enriched, risk_band)

        return PredictionResponse(
            model_name=selected_model_name,
            risk_probability=round(probability, 4),
            risk_percent=round(probability * 100, 2),
            risk_band=risk_band,
            prediction_confidence=round(prediction_confidence, 2),
            explanation=explanation,
            key_factors=factors,
            normalized_input=NormalizedFeatures(**enriched),
            missing_required_fields=missing,
            model_metrics=selected_metrics,
            recommendations=recommendations,
        )
