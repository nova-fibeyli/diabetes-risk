export type FeatureField = {
  key: string;
  label: string;
  type: "number" | "integer" | "string" | "boolean";
  required: boolean;
  step: number;
  section: string;
  min?: number;
  max?: number;
  description: string;
  options?: string[];
  improves_accuracy?: boolean;
};

export type FeatureSchemaResponse = {
  fields: FeatureField[];
  disclaimer: string;
};

export type ModelInfoResponse = {
  model_name: string;
  trained_at: string | null;
  target_column: string;
  feature_columns: string[];
  metrics: Record<string, number>;
  all_model_metrics: Record<string, Record<string, number>>;
  available_models: string[];
  dataset_rows?: number | null;
  disclaimer: string;
};

export type PredictionInput = {
  age?: number;
  sex?: "female" | "male" | "other";
  height_cm?: number;
  weight_kg?: number;
  bmi?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  heart_rate?: number;
  fasting_glucose_mg_dl?: number;
  fasting_glucose_mmol_l?: number;
  hba1c?: number;
  insulin?: number;
  homa_ir?: number;
  cholesterol_total?: number;
  physical_activity_days_per_week?: number;
  smoking?: boolean;
  family_history_diabetes?: boolean;
  hypertension?: boolean;
  cardiovascular_disease?: boolean;
  stroke_history?: boolean;
  pregnancy_count?: number;
  notes?: string;
  derived_model_features?: Record<string, number>;
};

export type PredictionResponse = {
  model_name: string;
  risk_probability: number;
  risk_percent: number;
  risk_band: "low" | "moderate" | "high" | "incomplete";
  prediction_confidence: number;
  explanation: string;
  key_factors: string[];
  normalized_input: PredictionInput;
  missing_required_fields: string[];
  model_metrics: Record<string, number>;
  recommendations: RecommendationItem[];
  saved_prediction_id?: number | null;
  disclaimer: string;
};

export type RecommendationItem = {
  title: string;
  rationale: string;
  priority: "high" | "medium" | "low";
  category: "lifestyle" | "monitoring" | "laboratory" | "medical_follow_up";
};

export type ParsedField = {
  key: string;
  label: string;
  value: number | string | null;
  confidence: number;
  source_text?: string | null;
  required_for_prediction: boolean;
};

export type ParsePdfResponse = {
  filename: string;
  extracted_text_preview: string;
  extracted_fields: ParsedField[];
  missing_required_fields: string[];
  uploaded_report_id?: number | null;
  disclaimer: string;
};

export type MetricsResponse = {
  prediction_requests: number;
  batch_requests: number;
  pdf_parse_requests: number;
  model_metrics: Record<string, number>;
  model_name: string;
};

export type AuthConfigResponse = {
  configured: boolean;
  login_url: string;
};

export type HistoryItem = {
  id: number;
  created_at: string;
  risk_percent: number;
  risk_band: string;
  prediction_confidence: number;
  explanation: string;
  key_metrics: Record<string, unknown>;
};

export type HistoryResponse = {
  items: HistoryItem[];
};

export type UserProfile = {
  id: number;
  email: string;
  full_name?: string | null;
  avatar_url?: string | null;
  joined_at: string;
  last_login_at: string;
  is_admin: boolean;
};

export type ProfileResponse = {
  user: UserProfile;
  recent_predictions: HistoryItem[];
  disclaimer: string;
};

export type AdminUserSummary = {
  id: number;
  email: string;
  full_name?: string | null;
  avatar_url?: string | null;
  joined_at: string;
  last_login_at: string;
  prediction_count: number;
  report_count: number;
  average_risk_percent?: number | null;
  average_confidence?: number | null;
  recent_predictions: HistoryItem[];
};

export type AdminOverviewResponse = {
  users: AdminUserSummary[];
  generated_at: string;
};
