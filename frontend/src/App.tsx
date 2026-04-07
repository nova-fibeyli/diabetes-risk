import {
  Alert,
  Avatar,
  Button,
  Card,
  Collapse,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  Layout,
  List,
  Progress,
  Segmented,
  Skeleton,
  Space,
  Steps,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
  message,
} from "antd";
import {
  ArrowRightOutlined,
  CheckCircleFilled,
  FilePdfOutlined,
  GoogleOutlined,
  HistoryOutlined,
  LineChartOutlined,
  LogoutOutlined,
  RadarChartOutlined,
  SafetyCertificateOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  getAdminOverview,
  getAuthConfig,
  getFeatureSchema,
  getGoogleLoginUrl,
  getHistory,
  getModelInfo,
  getProfile,
  logout,
  parsePdf,
  predict,
} from "./api";
import fishBackgroundVideo from "../assets/fish-swiming-background.mp4";
import type {
  AdminOverviewResponse,
  AuthConfigResponse,
  FeatureField,
  HistoryItem,
  ModelInfoResponse,
  ParsePdfResponse,
  PredictionInput,
  PredictionResponse,
  ProfileResponse,
  RecommendationItem,
} from "./types";

const { Header, Content } = Layout;
const { Title, Paragraph, Text } = Typography;

type AppTab = "overview" | "assessment" | "prediction" | "recommendations" | "history" | "admin";

const stepTitles = [
  "Basic info",
  "Body metrics",
  "Medical history",
  "Lab values",
  "Review + predict",
];

const stepFieldGroups: string[][] = [
  ["age", "sex", "family_history_diabetes", "smoking"],
  ["height_cm", "weight_kg", "systolic_bp", "diastolic_bp", "physical_activity_days_per_week"],
  ["hypertension", "cardiovascular_disease", "stroke_history"],
  [],
  [],
];

const initialValues: PredictionInput = {
  sex: "female",
  family_history_diabetes: false,
  smoking: false,
  hypertension: false,
  cardiovascular_disease: false,
  stroke_history: false,
  physical_activity_days_per_week: 3,
};

const modelProfiles: Record<string, { label: string; speed: string; note: string }> = {
  LogisticRegression: {
    label: "Logistic regression",
    speed: "Fast",
    note: "Strong baseline with high interpretability and rapid inference.",
  },
  RandomForest: {
    label: "Random forest",
    speed: "Medium",
    note: "Ensemble tree model that captures non-linear relations with moderate latency.",
  },
  HistGradientBoosting: {
    label: "Histogram gradient boosting",
    speed: "Medium-fast",
    note: "Boosted tree method that often improves accuracy on tabular data.",
  },
  MLPClassifier: {
    label: "Multilayer perceptron",
    speed: "Slowest",
    note: "Neural baseline with flexible boundaries but lower interpretability.",
  },
};

function computeBmi(heightCm?: number, weightKg?: number): number | undefined {
  if (!heightCm || !weightKg || heightCm <= 0) return undefined;
  const heightM = heightCm / 100;
  return Number((weightKg / (heightM * heightM)).toFixed(1));
}

function fieldLabel(fields: FeatureField[], key: string): string {
  return fields.find((field) => field.key === key)?.label ?? key;
}

function factorTone(riskBand?: string) {
  if (riskBand === "high") return "#ef4444";
  if (riskBand === "moderate") return "#f59e0b";
  return "#14b8a6";
}

function priorityColor(priority: RecommendationItem["priority"]) {
  if (priority === "high") return "red";
  if (priority === "medium") return "gold";
  return "cyan";
}

function formatMetric(value?: number): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

function modelLabel(name: string): string {
  return modelProfiles[name]?.label ?? name;
}

function modelSpeed(name: string): string {
  return modelProfiles[name]?.speed ?? "Medium";
}

function modelNote(name: string): string {
  return modelProfiles[name]?.note ?? "Validation metrics are available for comparison.";
}

function compositeScore(metrics: Record<string, number>): number {
  return ((metrics.roc_auc ?? 0) * 0.5) + ((metrics.f1 ?? 0) * 0.35) + ((metrics.accuracy ?? 0) * 0.15);
}

function BackgroundLayer() {
  return (
    <div className="background-layer" aria-hidden="true">
      <video className="background-video" autoPlay muted loop playsInline preload="auto">
        <source src={fishBackgroundVideo} type="video/mp4" />
      </video>
      <div className="background-video-tint" />
    </div>
  );
}

function App() {
  const [form] = Form.useForm<PredictionInput>();
  const [featureSchema, setFeatureSchema] = useState<FeatureField[]>([]);
  const [authConfig, setAuthConfig] = useState<AuthConfigResponse | null>(null);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [adminOverview, setAdminOverview] = useState<AdminOverviewResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [selectedModelName, setSelectedModelName] = useState("");
  const [currentStep, setCurrentStep] = useState(0);
  const [activeTab, setActiveTab] = useState<AppTab>("overview");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [pdfData, setPdfData] = useState<ParsePdfResponse | null>(null);
  const [bootLoading, setBootLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [authRedirecting, setAuthRedirecting] = useState(false);
  const [formSnapshot, setFormSnapshot] = useState<PredictionInput>(initialValues);

  const values = formSnapshot;
  const bmiValue = useMemo(
    () => computeBmi(values.height_cm, values.weight_kg) ?? values.bmi,
    [values.height_cm, values.weight_kg, values.bmi],
  );

  const modelCatalog = useMemo(() => {
    const entries = Object.entries(modelInfo?.all_model_metrics ?? {});
    return entries
      .map(([name, metrics]) => ({
        name,
        metrics,
        score: compositeScore(metrics),
      }))
      .sort((left, right) => right.score - left.score);
  }, [modelInfo]);

  const bestModelName = modelCatalog[0]?.name ?? modelInfo?.model_name ?? "";

  const averageRisk = useMemo(() => {
    if (!history.length) return null;
    const total = history.reduce((sum, item) => sum + item.risk_percent, 0);
    return (total / history.length).toFixed(1);
  }, [history]);

  const averageConfidence = useMemo(() => {
    if (!history.length) return null;
    const total = history.reduce((sum, item) => sum + item.prediction_confidence, 0);
    return (total / history.length).toFixed(1);
  }, [history]);

  useEffect(() => {
    form.setFieldsValue(initialValues);
    setFormSnapshot(initialValues);
  }, [form]);

  useEffect(() => {
    const load = async () => {
      try {
        const [schemaResult, authResult, modelResult] = await Promise.allSettled([
          getFeatureSchema(),
          getAuthConfig(),
          getModelInfo(),
        ]);

        if (schemaResult.status === "fulfilled") {
          setFeatureSchema(schemaResult.value.fields);
        } else {
          message.error("The clinical form schema could not be loaded from the API.");
        }

        if (authResult.status === "fulfilled") {
          setAuthConfig(authResult.value);
        } else {
          message.error("Authentication status could not be loaded from the API.");
        }

        if (modelResult.status === "fulfilled") {
          setModelInfo(modelResult.value);
          setSelectedModelName(modelResult.value.model_name);
        } else {
          message.error("Model metadata could not be loaded from the API.");
        }

        try {
          const [profileResponse, historyResponse] = await Promise.all([getProfile(), getHistory()]);
          setProfile(profileResponse);
          setHistory(historyResponse.items);
          if (profileResponse.user.is_admin) {
            try {
              const adminResponse = await getAdminOverview();
              setAdminOverview(adminResponse);
            } catch {
              setAdminOverview(null);
            }
          }
        } catch {
          setProfile(null);
          setHistory([]);
          setAdminOverview(null);
        }
      } catch (error) {
        if (error instanceof Error) message.error(error.message);
      } finally {
        setBootLoading(false);
      }
    };
    void load();
  }, []);

  const requiredMissing = useMemo(() => {
    return featureSchema
      .filter((field) => field.required)
      .filter((field) => {
        const value = (values as Record<string, unknown>)[field.key];
        return value === undefined || value === null || value === "";
      })
      .map((field) => field.key);
  }, [featureSchema, values]);

  const recentPredictions = profile?.recent_predictions ?? [];
  const historyBars = history.slice(0, 8).reverse();

  const syncFormSnapshot = () => {
    setFormSnapshot(form.getFieldsValue(true) as PredictionInput);
  };

  const goToNextStep = () => {
    const requiredForStep = stepFieldGroups[currentStep] ?? [];
    const missingForStep = requiredForStep.filter((key) => requiredMissing.includes(key));
    if (missingForStep.length > 0) {
      message.warning(
        `Complete this step first: ${missingForStep.map((key) => fieldLabel(featureSchema, key)).join(", ")}`,
      );
      return;
    }
    setCurrentStep((step) => Math.min(step + 1, stepTitles.length - 1));
  };

  const goToPreviousStep = () => setCurrentStep((step) => Math.max(step - 1, 0));

  const startGoogleLogin = () => {
    setAuthRedirecting(true);
    const target = authConfig?.login_url || getGoogleLoginUrl();
    window.setTimeout(() => {
      window.location.href = target;
    }, 100);
  };

  const refreshHistory = async () => {
    try {
      const [profileResponse, historyResponse] = await Promise.all([getProfile(), getHistory()]);
      setProfile(profileResponse);
      setHistory(historyResponse.items);
      if (profileResponse.user.is_admin) {
        const adminResponse = await getAdminOverview();
        setAdminOverview(adminResponse);
      } else {
        setAdminOverview(null);
      }
    } catch (error) {
      if (error instanceof Error) message.error(error.message);
    }
  };

  const signOut = async () => {
    await logout();
    setProfile(null);
    setHistory([]);
    setAdminOverview(null);
    setResult(null);
    setActiveTab("overview");
  };

  const submitPrediction = async () => {
    if (requiredMissing.length > 0) {
      message.warning(
        `Some required fields are still missing: ${requiredMissing.map((key) => fieldLabel(featureSchema, key)).join(", ")}`,
      );
      return;
    }

    try {
      setAnalyzing(true);
      const input = form.getFieldsValue(true) as PredictionInput;
      input.bmi = computeBmi(input.height_cm, input.weight_kg) ?? input.bmi;
      setFormSnapshot(input);
      const prediction = await predict(input, selectedModelName || undefined);
      setResult(prediction);
      setActiveTab("prediction");
      await refreshHistory();
    } catch (error) {
      if (error instanceof Error) message.error(error.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handlePdfUpload = async (file: File) => {
    setPdfLoading(true);
    try {
      const parsed = await parsePdf(file);
      setPdfData(parsed);
      const nextValues: PredictionInput = {};
      parsed.extracted_fields.forEach((field) => {
        if (typeof field.value === "number") {
          nextValues[field.key as keyof PredictionInput] = field.value as never;
        }
      });
      form.setFieldsValue(nextValues);
      syncFormSnapshot();
      message.success(`Imported values from ${file.name}`);
    } catch (error) {
      if (error instanceof Error) message.error(error.message);
    } finally {
      setPdfLoading(false);
    }
    return false;
  };

  const reviewItems = [
    ["Age", values.age],
    ["Sex", values.sex],
    ["Height", values.height_cm ? `${values.height_cm} cm` : null],
    ["Weight", values.weight_kg ? `${values.weight_kg} kg` : null],
    ["BMI", bmiValue],
    ["Blood pressure", values.systolic_bp && values.diastolic_bp ? `${values.systolic_bp}/${values.diastolic_bp}` : null],
    ["Physical activity", values.physical_activity_days_per_week !== undefined ? `${values.physical_activity_days_per_week} day(s) per week` : null],
    ["Fasting glucose", values.fasting_glucose_mg_dl ? `${values.fasting_glucose_mg_dl} mg/dL` : null],
    ["HbA1c", values.hba1c ? `${values.hba1c}%` : null],
    ["Insulin", values.insulin ? `${values.insulin} uIU/mL` : null],
    ["HOMA-IR", values.homa_ir],
  ].filter((item) => item[1] !== null && item[1] !== undefined);

  const renderLoadingOverlay = () => {
    if (!analyzing && !authRedirecting) return null;

    return (
      <AnimatePresence>
        <motion.div
          className="loading-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="loading-dialog"
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
          >
            <div className="loading-dots" aria-hidden="true">
              <span className="loading-dot" />
              <span className="loading-dot" />
              <span className="loading-dot" />
            </div>
            <Title level={4} style={{ marginBottom: 8 }}>
              {authRedirecting ? "Opening research session" : "Running selected model"}
            </Title>
            <Paragraph className="muted" style={{ marginBottom: 0 }}>
              {authRedirecting
                ? "Redirecting to Google OAuth for session-based access."
                : "Preparing normalized features, executing inference, and collecting the confidence estimate."}
            </Paragraph>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    );
  };

  const renderModelChoiceGrid = () => {
    if (!modelCatalog.length) {
      return (
        <Alert
          type="info"
          showIcon
          message="Model comparison is unavailable"
          description="The backend did not return per-model validation metrics for this run."
        />
      );
    }

    return (
      <div className="model-choice-grid">
        {modelCatalog.map((item) => (
          <button
            type="button"
            key={item.name}
            className={`model-choice-card ${selectedModelName === item.name ? "active" : ""}`}
            onClick={() => setSelectedModelName(item.name)}
          >
            <div className="model-choice-head">
              <div>
                <strong>{modelLabel(item.name)}</strong>
                <Text className="muted model-subline">{modelNote(item.name)}</Text>
              </div>
              <Space size={6} wrap>
                {item.name === bestModelName ? <Tag color="cyan">Best validation score</Tag> : null}
                {selectedModelName === item.name ? <Tag color="green">Selected</Tag> : null}
              </Space>
            </div>
            <div className="model-choice-meta">
              <span>{modelSpeed(item.name)}</span>
              <span>ROC-AUC {formatMetric(item.metrics.roc_auc)}</span>
            </div>
            <div className="metric-stack">
              {[
                ["Accuracy", item.metrics.accuracy],
                ["F1", item.metrics.f1],
                ["ROC-AUC", item.metrics.roc_auc],
              ].map(([label, value]) => (
                <div key={label} className="metric-bar-row">
                  <span>{label}</span>
                  <div className="metric-bar-track">
                    <div className="metric-bar-fill" style={{ width: `${Math.max(((value as number) ?? 0) * 100, 4)}%` }} />
                  </div>
                  <strong>{formatMetric(value as number)}</strong>
                </div>
              ))}
            </div>
          </button>
        ))}
      </div>
    );
  };

  const renderAuth = () => (
    <>
      <BackgroundLayer />
      <Layout className="auth-shell">
        <Content className="auth-content">
          <div className="auth-stage">
            <section className="auth-hero">
            <Tag className="accent-pill" icon={<SafetyCertificateOutlined />}>
              Research Interface
            </Tag>
            <Title className="auth-title">
              Type 2 diabetes risk estimation interface for diploma research, structured data entry, and comparative model evaluation.
            </Title>
            <Paragraph className="auth-copy">
              The interface is organized as a stepwise protocol: demographic data, anthropometric measurements,
              cardiovascular history, and optional laboratory markers. PDF laboratory reports can be parsed and then
              corrected manually before inference.
            </Paragraph>
            <div className="auth-feature-grid">
              <div className="auth-feature-card"><RadarChartOutlined /><span>Stepwise variable entry</span></div>
              <div className="auth-feature-card"><FilePdfOutlined /><span>Editable laboratory extraction</span></div>
              <div className="auth-feature-card"><HistoryOutlined /><span>Session-linked run history</span></div>
            </div>
            </section>

            <Card className="auth-panel" bordered={false}>
              <Space direction="vertical" size="large" style={{ width: "100%" }}>
              <div>
                <Text className="eyebrow">Session Sign-In</Text>
                <Title level={3}>Open the saved research workspace</Title>
                <Paragraph className="muted">
                  Google OAuth is used only for restoring saved runs, uploaded reports, and repeated experimental
                  sessions. The predictive output remains a support estimate and not a diagnosis.
                </Paragraph>
              </div>
              <Button
                type="primary"
                size="large"
                icon={<GoogleOutlined />}
                block
                className="google-auth-button"
                onClick={startGoogleLogin}
                disabled={authConfig?.configured !== true}
              >
                Continue with Google
              </Button>
              {authConfig === null ? (
                <Alert
                  type="error"
                  showIcon
                  message="Backend status unavailable"
                  description="The API did not return authentication status. Check the backend container logs first."
                />
              ) : !authConfig.configured ? (
                <Alert
                  type="warning"
                  showIcon
                  message="Google OAuth is not configured yet"
                  description="Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the backend environment to enable sign-in."
                />
              ) : (
                <Alert
                  type="info"
                  showIcon
                  message="Scientific workflow"
                  description={`Validated models available: ${modelInfo?.available_models?.length ?? modelCatalog.length}. The default benchmark is ${modelLabel(selectedModelName || bestModelName || modelInfo?.model_name || "the current model")}.`}
                />
              )}
              </Space>
            </Card>
          </div>
        </Content>
        {renderLoadingOverlay()}
      </Layout>
    </>
  );

  const renderOverview = () => (
    <motion.div key="overview" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      <section className="hero-panel">
        <div className="hero-copy">
          <Tag className="accent-pill" icon={<LineChartOutlined />}>Study overview</Tag>
          <Title className="hero-title">
            Structured diabetes risk estimation with repeatable input protocol, optional laboratory enrichment, and comparative model selection.
          </Title>
          <Paragraph className="hero-text">
            This interface was reframed for diploma work rather than product marketing. Each run stores the entered
            variables, selected model, estimated probability, and confidence value so repeated experiments can be
            compared across time and across classifiers.
          </Paragraph>
          <Space wrap>
            <Button type="primary" size="large" onClick={() => setActiveTab("assessment")}>Start new run</Button>
            <Button size="large" onClick={() => setActiveTab("history")}>Open run history</Button>
          </Space>
        </div>
        <div className="hero-stats">
          <div className="metric-tile">
            <span className="metric-label">Stored runs</span>
            <strong>{history.length}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Average estimated risk</span>
            <strong>{averageRisk ? `${averageRisk}%` : "No data"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Preferred benchmark</span>
            <strong>{modelLabel(selectedModelName || bestModelName || modelInfo?.model_name || "Not loaded")}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Average confidence</span>
            <strong>{averageConfidence ? `${averageConfidence}%` : "No data"}</strong>
          </div>
        </div>
      </section>

      <section className="insight-grid">
        <Card className="insight-card" bordered={false}>
          <Title level={4}>Protocol structure</Title>
          <Paragraph className="muted">
            The workflow separates demographic context, body metrics, medical history, and laboratory variables to
            reduce clutter and make the entered factors easier to audit.
          </Paragraph>
        </Card>
        <Card className="insight-card accent-card" bordered={false}>
          <Title level={4}>Laboratory import</Title>
          <Paragraph className="muted">
            Text-based reports can be parsed for glucose, insulin, HbA1c, HOMA-IR, and cholesterol, then corrected
            manually before the run is submitted.
          </Paragraph>
        </Card>
        <Card className="insight-card" bordered={false}>
          <Title level={4}>Comparative models</Title>
          <Paragraph className="muted">
            Multiple candidate classifiers are available. Validation metrics are shown explicitly so model selection is
            transparent rather than hidden behind a single default score.
          </Paragraph>
        </Card>
      </section>

      <section className="overview-secondary-grid">
        <Card className="history-preview-card" bordered={false}>
          <div className="section-header">
            <div>
              <Title level={4}>Candidate models</Title>
              <Paragraph className="muted">
                Validation metrics are shown from the training pipeline. Select a model here or inside the review step.
              </Paragraph>
            </div>
          </div>
          {renderModelChoiceGrid()}
        </Card>

        <Card className="history-preview-card" bordered={false}>
          <div className="section-header">
            <div>
              <Title level={4}>Recent estimated risk values</Title>
              <Paragraph className="muted">
                Simple bar view of the latest saved runs for quick comparison.
              </Paragraph>
            </div>
          </div>
          {historyBars.length ? (
            <div className="history-bars">
              {historyBars.map((item) => (
                <div key={item.id} className="history-bar-item">
                  <div className="history-bar-track">
                    <motion.div
                      className="history-bar-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(item.risk_percent, 4)}%` }}
                      transition={{ duration: 0.45 }}
                      style={{ background: factorTone(item.risk_band) }}
                    />
                  </div>
                  <div className="history-bar-meta">
                    <span>{new Date(item.created_at).toLocaleDateString()}</span>
                    <strong>{item.risk_percent}%</strong>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Alert
              type="info"
              showIcon
              message="No saved runs yet"
              description="After the first prediction, this area will display a compact comparison of recent estimated risk values."
            />
          )}
        </Card>
      </section>

      <Card className="history-preview-card" bordered={false}>
        <div className="section-header">
          <Title level={4}>Recent assessments</Title>
          <Button type="link" onClick={() => setActiveTab("history")}>See full history</Button>
        </div>
        {recentPredictions.length ? (
          <div className="history-preview-grid">
            {recentPredictions.map((item) => (
              <motion.div key={item.id} whileHover={{ y: -3 }} transition={{ duration: 0.2 }} className="history-preview-item">
                <span className="history-date">{new Date(item.created_at).toLocaleString()}</span>
                <strong>{item.risk_percent}%</strong>
                <Tag color={item.risk_band === "high" ? "red" : item.risk_band === "moderate" ? "gold" : "cyan"}>
                  {item.risk_band.toUpperCase()}
                </Tag>
                {typeof item.key_metrics.model_name === "string" ? (
                  <Text className="muted">{modelLabel(item.key_metrics.model_name)}</Text>
                ) : null}
                <Text className="muted">Confidence {item.prediction_confidence}%</Text>
              </motion.div>
            ))}
          </div>
        ) : (
          <Alert
            type="info"
            showIcon
            message="No stored assessments yet"
            description="Run the protocol once to populate saved results, confidence values, and comparative history."
          />
        )}
      </Card>
    </motion.div>
  );

  const renderAssessment = () => (
    <motion.div key="assessment" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      <section className="assessment-shell">
        <Card className="wizard-card" bordered={false}>
          <div className="section-header">
            <div>
              <Text className="eyebrow">Assessment Protocol</Text>
              <Title level={3}>New experimental run</Title>
            </div>
            <Tag className="accent-pill">Step {currentStep + 1} of {stepTitles.length}</Tag>
          </div>

          <Steps
            current={currentStep}
            onChange={setCurrentStep}
            responsive
            items={stepTitles.map((title) => ({ title }))}
            className="wizard-steps clickable-steps"
          />

          <Form<PredictionInput>
            form={form}
            layout="vertical"
            onValuesChange={(changedValues, allValues) => {
              if (changedValues.sex === "male" && allValues.pregnancy_count !== undefined) {
                form.setFieldValue("pregnancy_count", undefined);
                setFormSnapshot({
                  ...(allValues as PredictionInput),
                  pregnancy_count: undefined,
                });
                return;
              }
              setFormSnapshot(form.getFieldsValue(true) as PredictionInput);
            }}
          >
            <AnimatePresence mode="wait">
              <motion.div key={currentStep} initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -18 }} transition={{ duration: 0.24 }}>
                {currentStep === 0 && (
                  <div className="step-grid">
                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Basic variables</Title>
                      <Paragraph className="muted">
                        Enter demographic variables required for all runs. These are mandatory for inference.
                      </Paragraph>
                      <Form.Item label="Age" name="age" required>
                        <InputNumber size="large" min={18} max={120} style={{ width: "100%" }} />
                      </Form.Item>
                      <Form.Item label="Sex" name="sex" required>
                        <Segmented
                          block
                          options={[
                            { label: "Female", value: "female" },
                            { label: "Male", value: "male" },
                            { label: "Other", value: "other" },
                          ]}
                        />
                      </Form.Item>
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Risk context</Title>
                      <Paragraph className="muted">
                        These binary variables provide baseline epidemiological and behavioral context.
                      </Paragraph>
                      <Form.Item label="Family history of diabetes" name="family_history_diabetes" required>
                        <Segmented block options={[{ label: "No", value: false }, { label: "Yes", value: true }]} />
                      </Form.Item>
                      <Form.Item label="Smoking" name="smoking" required>
                        <Segmented block options={[{ label: "No", value: false }, { label: "Yes", value: true }]} />
                      </Form.Item>
                    </Card>
                  </div>
                )}

                {currentStep === 1 && (
                  <div className="step-grid">
                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Anthropometric measurements</Title>
                      <Form.Item label="Height (cm)" name="height_cm" required>
                        <InputNumber size="large" min={100} max={250} style={{ width: "100%" }} />
                      </Form.Item>
                      <Form.Item label="Weight (kg)" name="weight_kg" required>
                        <InputNumber size="large" min={20} max={350} style={{ width: "100%" }} />
                      </Form.Item>
                      <div className="inline-stat">
                        <span className="metric-label">Calculated BMI</span>
                        <strong>{bmiValue ?? "Pending"}</strong>
                      </div>
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Cardiovascular measurements</Title>
                      <div className="split-inputs">
                        <Form.Item label="Systolic BP" name="systolic_bp" required>
                          <InputNumber size="large" min={70} max={250} style={{ width: "100%" }} />
                        </Form.Item>
                        <Form.Item label="Diastolic BP" name="diastolic_bp" required>
                          <InputNumber size="large" min={40} max={150} style={{ width: "100%" }} />
                        </Form.Item>
                      </div>
                      <div className="split-inputs">
                        <Form.Item label="Heart rate" name="heart_rate">
                          <InputNumber size="large" min={30} max={220} style={{ width: "100%" }} />
                        </Form.Item>
                        <Form.Item label="Physical activity days / week" name="physical_activity_days_per_week" required>
                          <InputNumber size="large" min={0} max={7} style={{ width: "100%" }} />
                        </Form.Item>
                      </div>
                    </Card>
                  </div>
                )}

                {currentStep === 2 && (
                  <div className="step-grid">
                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Medical history</Title>
                      <Paragraph className="muted">
                        Binary clinical history variables used in the current experimental protocol.
                      </Paragraph>
                      <Form.Item label="Hypertension" name="hypertension" required>
                        <Segmented block options={[{ label: "No", value: false }, { label: "Yes", value: true }]} />
                      </Form.Item>
                      <Form.Item label="Cardiovascular disease" name="cardiovascular_disease" required>
                        <Segmented block options={[{ label: "No", value: false }, { label: "Yes", value: true }]} />
                      </Form.Item>
                      <Form.Item label="Stroke history" name="stroke_history" required>
                        <Segmented block options={[{ label: "No", value: false }, { label: "Yes", value: true }]} />
                      </Form.Item>
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Optional modifiers</Title>
                      <Paragraph className="muted">
                        Notes remain optional. Pregnancy count is shown only when the selected sex is not male, to keep the
                        protocol semantically consistent.
                      </Paragraph>
                      {values.sex !== "male" ? (
                        <Form.Item label="Pregnancy count" name="pregnancy_count">
                          <InputNumber size="large" min={0} max={20} style={{ width: "100%" }} />
                        </Form.Item>
                      ) : null}
                      <Form.Item label="Notes" name="notes">
                        <Input.TextArea rows={5} placeholder="Optional remarks about this run, report source, or laboratory context." />
                      </Form.Item>
                    </Card>
                  </div>
                )}

                {currentStep === 3 && (
                  <div className="step-grid labs-grid">
                    <Card className="field-card" bordered={false}>
                      <div className="section-header">
                        <div>
                          <Title level={4}>Laboratory variables</Title>
                          <Paragraph className="muted">
                            Optional inputs. When present, they increase interpretability and usually improve confidence.
                          </Paragraph>
                        </div>
                        <Tag color="cyan">Optional but useful</Tag>
                      </div>
                      <Collapse
                        defaultActiveKey={["labs"]}
                        ghost
                        items={[
                          {
                            key: "labs",
                            label: "Enter laboratory values manually",
                            children: (
                              <div className="step-grid compact-grid">
                                <Form.Item label="Fasting glucose (mg/dL)" name="fasting_glucose_mg_dl">
                                  <InputNumber size="large" min={40} max={500} style={{ width: "100%" }} />
                                </Form.Item>
                                <Form.Item label="Fasting glucose (mmol/L)" name="fasting_glucose_mmol_l">
                                  <InputNumber size="large" min={2} max={30} style={{ width: "100%" }} />
                                </Form.Item>
                                <Form.Item label="HbA1c (%)" name="hba1c">
                                  <InputNumber size="large" min={3} max={20} style={{ width: "100%" }} />
                                </Form.Item>
                                <Form.Item label="Insulin (uIU/mL)" name="insulin">
                                  <InputNumber size="large" min={0} max={1000} style={{ width: "100%" }} />
                                </Form.Item>
                                <Form.Item label="HOMA-IR" name="homa_ir">
                                  <InputNumber size="large" min={0} max={30} style={{ width: "100%" }} />
                                </Form.Item>
                                <Form.Item label="Total cholesterol (mg/dL)" name="cholesterol_total">
                                  <InputNumber size="large" min={50} max={500} style={{ width: "100%" }} />
                                </Form.Item>
                              </div>
                            ),
                          },
                        ]}
                      />
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Laboratory PDF import</Title>
                      <Paragraph className="muted">
                        Upload a text-based report to extract glucose, insulin, HOMA-IR, HbA1c, and cholesterol fields.
                      </Paragraph>
                      <Upload.Dragger beforeUpload={handlePdfUpload} showUploadList={false} accept=".pdf" className="pdf-dropzone">
                        <p className="ant-upload-drag-icon"><UploadOutlined /></p>
                        <p className="ant-upload-text">Drop a report here or tap to upload</p>
                        <p className="ant-upload-hint">Extracted values remain editable inside the form.</p>
                      </Upload.Dragger>
                      {pdfLoading ? (
                        <Skeleton active paragraph={{ rows: 3 }} />
                      ) : pdfData ? (
                        <Alert
                          type={pdfData.extracted_fields.length ? "success" : "warning"}
                          showIcon
                          message={pdfData.extracted_fields.length ? "Laboratory values detected" : "No numeric laboratory values detected"}
                          description={pdfData.extracted_text_preview.slice(0, 180) || "The file was opened but no text-based values were captured."}
                        />
                      ) : null}
                    </Card>

                    <Card className="field-card wide-card" bordered={false}>
                      <Title level={4}>Extracted report values</Title>
                      <Table
                        pagination={false}
                        size="small"
                        rowKey="key"
                        scroll={{ x: 420 }}
                        dataSource={pdfData?.extracted_fields ?? []}
                        locale={{ emptyText: "Upload a PDF to populate editable laboratory values." }}
                        columns={[
                          { title: "Field", dataIndex: "label", key: "label" },
                          { title: "Value", dataIndex: "value", key: "value" },
                          {
                            title: "Confidence",
                            dataIndex: "confidence",
                            key: "confidence",
                            render: (value: number) => `${Math.round(value * 100)}%`,
                          },
                          {
                            title: "Matched text",
                            dataIndex: "source_text",
                            key: "source_text",
                            render: (value: string | null) => value ?? "n/a",
                          },
                        ]}
                      />
                    </Card>
                  </div>
                )}

                {currentStep === 4 && (
                  <div className="review-stack">
                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Model selection</Title>
                      <Paragraph className="muted">
                        Choose the classifier for this run. Validation metrics are shown to support comparison between
                        speed and accuracy trade-offs.
                      </Paragraph>
                      {renderModelChoiceGrid()}
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Review before prediction</Title>
                      <Paragraph className="muted">
                        This summary uses the values currently preserved in the form. Missing laboratory variables are
                        allowed; missing required intake variables are not.
                      </Paragraph>
                      <Descriptions column={1} size="small" bordered>
                        {reviewItems.map(([label, value]) => (
                          <Descriptions.Item key={label} label={label}>
                            {String(value)}
                          </Descriptions.Item>
                        ))}
                        <Descriptions.Item label="Selected model">
                          {modelLabel(selectedModelName || bestModelName || modelInfo?.model_name || "Not selected")}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>

                    <Card className="field-card" bordered={false}>
                      <Title level={4}>Readiness</Title>
                      <Alert
                        type={requiredMissing.length ? "warning" : "success"}
                        showIcon
                        message={
                          requiredMissing.length
                            ? `Missing required fields: ${requiredMissing.map((key) => fieldLabel(featureSchema, key)).join(", ")}`
                            : "All required protocol fields are available."
                        }
                      />
                      <Divider />
                      <Button
                        type="primary"
                        size="large"
                        block
                        className="analyze-button"
                        loading={analyzing}
                        onClick={submitPrediction}
                      >
                        {analyzing ? "Running selected model..." : "Generate diabetes risk estimate"}
                      </Button>
                      <Paragraph className="muted tiny-copy">
                        This output is a scientific support estimate only and must not be interpreted as a clinical diagnosis.
                      </Paragraph>
                    </Card>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </Form>

          <div className="wizard-footer">
            <Button size="large" onClick={goToPreviousStep} disabled={currentStep === 0}>
              Back
            </Button>
            {currentStep < stepTitles.length - 1 ? (
              <Button type="primary" size="large" icon={<ArrowRightOutlined />} iconPosition="end" onClick={goToNextStep}>
                Continue
              </Button>
            ) : null}
          </div>
        </Card>

        <Card className="assistant-card" bordered={false}>
          <Title level={4}>Assessment guidance</Title>
          <List
            dataSource={[
              "Required variables can be entered in any order because the step bar is directly clickable.",
              "BMI is derived automatically from height and weight.",
              "Optional laboratory variables improve contextual quality and usually increase confidence.",
              "The selected model is attached to the saved run for later comparison.",
            ]}
            renderItem={(item) => (
              <List.Item>
                <Space>
                  <CheckCircleFilled style={{ color: "#14b8a6" }} />
                  <span>{item}</span>
                </Space>
              </List.Item>
            )}
          />
        </Card>
      </section>
    </motion.div>
  );

  const renderPrediction = () => (
    <motion.div key="prediction" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      {result ? (
        <>
          <section className="result-hero" style={{ ["--riskColor" as never]: factorTone(result.risk_band) }}>
            <div className="result-summary">
              <Tag className="accent-pill">{result.risk_band.toUpperCase()} RISK</Tag>
              <Title className="result-title">{result.risk_percent}% estimated risk</Title>
              <Paragraph className="result-text">{result.explanation}</Paragraph>
              <Space wrap>
                <Tag color="cyan">{modelLabel(result.model_name)}</Tag>
                <Tag color="default">Confidence {result.prediction_confidence}%</Tag>
                <Tag color="default">Saved run #{result.saved_prediction_id}</Tag>
                <Button size="small" onClick={() => setActiveTab("recommendations")}>Open recommendations</Button>
              </Space>
            </div>
            <motion.div className="result-gauge" initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.35, delay: 0.1 }}>
              <Progress
                type="dashboard"
                percent={result.risk_percent}
                strokeColor={factorTone(result.risk_band)}
                trailColor="#e5eef0"
                format={(percent) => `${percent}%`}
              />
            </motion.div>
          </section>

          <section className="result-grid">
            <Card className="result-card" bordered={false}>
              <Title level={4}>Primary contributing factors</Title>
              <List
                dataSource={result.key_factors}
                renderItem={(item) => (
                  <List.Item>
                    <Space>
                      <span className="factor-dot" style={{ background: factorTone(result.risk_band) }} />
                      <span>{item}</span>
                    </Space>
                  </List.Item>
                )}
              />
            </Card>

            <Card className="result-card" bordered={false}>
              <Title level={4}>Input snapshot</Title>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="Age">{result.normalized_input.age}</Descriptions.Item>
                <Descriptions.Item label="BMI">{result.normalized_input.bmi}</Descriptions.Item>
                <Descriptions.Item label="Blood pressure">
                  {result.normalized_input.systolic_bp}/{result.normalized_input.diastolic_bp}
                </Descriptions.Item>
                <Descriptions.Item label="Fasting glucose">{result.normalized_input.fasting_glucose_mg_dl ?? "Not provided"}</Descriptions.Item>
                <Descriptions.Item label="HbA1c">{result.normalized_input.hba1c ?? "Not provided"}</Descriptions.Item>
                <Descriptions.Item label="HOMA-IR">{result.normalized_input.homa_ir ?? "Not provided"}</Descriptions.Item>
              </Descriptions>
            </Card>

            <Card className="result-card result-card-wide" bordered={false}>
              <Title level={4}>Selected model performance context</Title>
              <Paragraph className="muted">
                Validation metrics below describe the selected classifier from the training pipeline rather than the current individual record.
              </Paragraph>
              <div className="metric-stack">
                {[
                  ["Accuracy", result.model_metrics.accuracy],
                  ["Precision", result.model_metrics.precision],
                  ["Recall", result.model_metrics.recall],
                  ["F1", result.model_metrics.f1],
                  ["ROC-AUC", result.model_metrics.roc_auc],
                ].map(([label, value]) => (
                  <div key={label} className="metric-bar-row">
                    <span>{label}</span>
                    <div className="metric-bar-track">
                      <div className="metric-bar-fill" style={{ width: `${Math.max(((value as number) ?? 0) * 100, 4)}%` }} />
                    </div>
                    <strong>{formatMetric(value as number)}</strong>
                  </div>
                ))}
              </div>
            </Card>
          </section>

          <Card className="history-preview-card" bordered={false}>
            <div className="section-header">
              <div>
                <Title level={4}>Model comparison for this study</Title>
                <Paragraph className="muted">
                  Use the model catalogue to compare the selected classifier against the remaining candidates.
                </Paragraph>
              </div>
              <Button onClick={() => setActiveTab("assessment")}>Change model and rerun</Button>
            </div>
            {renderModelChoiceGrid()}
          </Card>

          <Alert type="info" showIcon message="Research disclaimer" description={result.disclaimer} />
        </>
      ) : (
        <Card className="empty-state-card" bordered={false}>
          <Title level={4}>No prediction yet</Title>
          <Paragraph className="muted">
            Complete the protocol and execute one of the available models to populate this section.
          </Paragraph>
          <Button type="primary" onClick={() => setActiveTab("assessment")}>Open assessment protocol</Button>
        </Card>
      )}
    </motion.div>
  );

  const renderRecommendations = () => (
    <motion.div key="recommendations" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      {result ? (
        <>
          <Card className="history-preview-card" bordered={false}>
            <div className="section-header">
              <div>
                <Text className="eyebrow">Recommendation Summary</Text>
                <Title level={3}>Recommendations derived from current factors</Title>
              </div>
              <Tag color="cyan">{result.recommendations.length} items</Tag>
            </div>
            <Paragraph className="muted">
              These follow-up suggestions are rule-based and are generated from the entered measurements, history fields,
              and any imported laboratory values. They are meant for research interpretation, not direct medical advice.
            </Paragraph>
          </Card>

          <div className="recommendations-grid">
            {result.recommendations.map((item) => (
              <Card key={`${item.category}-${item.title}`} className="result-card recommendation-card" bordered={false}>
                <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                  <div className="section-header">
                    <Title level={4} style={{ marginBottom: 0 }}>{item.title}</Title>
                    <Tag color={priorityColor(item.priority)}>{item.priority.toUpperCase()}</Tag>
                  </div>
                  <Tag>{item.category.split("_").join(" ")}</Tag>
                  <Paragraph className="muted" style={{ marginBottom: 0 }}>
                    {item.rationale}
                  </Paragraph>
                </Space>
              </Card>
            ))}
          </div>
        </>
      ) : (
        <Card className="empty-state-card" bordered={false}>
          <Title level={4}>No recommendations yet</Title>
          <Paragraph className="muted">
            Run a prediction first to generate factor-based recommendations such as smoking cessation, exercise increase,
            repeat blood tests, or weight management follow-up.
          </Paragraph>
          <Button type="primary" onClick={() => setActiveTab("assessment")}>Open assessment protocol</Button>
        </Card>
      )}
    </motion.div>
  );

  const renderAdmin = () => (
    <motion.div key="admin" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      <Card className="history-page-card" bordered={false}>
        <div className="section-header">
          <div>
            <Text className="eyebrow">Admin Panel</Text>
            <Title level={3}>Google account activity and saved histories</Title>
          </div>
          <Tag color="cyan">{adminOverview?.users.length ?? 0} account(s)</Tag>
        </div>
        <Paragraph className="muted">
          This view aggregates Google-linked users, their saved prediction counts, uploaded report counts, and recent
          runs. Use it to review activity across accounts during testing or supervision.
        </Paragraph>
      </Card>

      <Card className="history-page-card" bordered={false}>
        <Table
          rowKey="id"
          pagination={false}
          scroll={{ x: 900 }}
          dataSource={adminOverview?.users ?? []}
          locale={{ emptyText: "No admin data available yet. Add your email to ADMIN_EMAILS and save some runs." }}
          columns={[
            {
              title: "Google account",
              key: "account",
              render: (_, item) => (
                <div className="admin-account-cell">
                  <Avatar src={item.avatar_url || undefined}>
                    {(item.full_name || item.email).slice(0, 1).toUpperCase()}
                  </Avatar>
                  <div className="admin-account-text">
                    <strong>{item.full_name || "Unnamed user"}</strong>
                    <span>{item.email}</span>
                  </div>
                </div>
              ),
            },
            { title: "Runs", dataIndex: "prediction_count", key: "prediction_count" },
            { title: "Reports", dataIndex: "report_count", key: "report_count" },
            {
              title: "Avg risk",
              key: "average_risk_percent",
              render: (_, item) => (item.average_risk_percent !== null && item.average_risk_percent !== undefined ? `${item.average_risk_percent}%` : "n/a"),
            },
            {
              title: "Avg confidence",
              key: "average_confidence",
              render: (_, item) => (item.average_confidence !== null && item.average_confidence !== undefined ? `${item.average_confidence}%` : "n/a"),
            },
            {
              title: "Last login",
              dataIndex: "last_login_at",
              key: "last_login_at",
              render: (value: string) => new Date(value).toLocaleString(),
            },
          ]}
          expandable={{
            expandedRowRender: (item) => (
              <div className="admin-history-stack">
                {item.recent_predictions.length ? item.recent_predictions.map((entry) => (
                  <div key={entry.id} className="admin-history-item">
                    <div>
                      <strong>{entry.risk_percent}%</strong>
                      <Text className="muted"> {new Date(entry.created_at).toLocaleString()}</Text>
                    </div>
                    <div className="admin-history-tags">
                      <Tag color={entry.risk_band === "high" ? "red" : entry.risk_band === "moderate" ? "gold" : "cyan"}>
                        {entry.risk_band.toUpperCase()}
                      </Tag>
                      {typeof entry.key_metrics.model_name === "string" ? <Tag>{modelLabel(entry.key_metrics.model_name)}</Tag> : null}
                      <Tag>Confidence {entry.prediction_confidence}%</Tag>
                    </div>
                    <Paragraph className="muted" style={{ marginBottom: 0 }}>{entry.explanation}</Paragraph>
                  </div>
                )) : (
                  <Text className="muted">No stored runs for this account yet.</Text>
                )}
              </div>
            ),
          }}
        />
      </Card>
    </motion.div>
  );

  const renderHistory = () => (
    <motion.div key="history" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="page-stack">
      <Card className="history-page-card" bordered={false}>
        <div className="section-header">
          <div>
            <Text className="eyebrow">Stored Runs</Text>
            <Title level={3}>Prediction history</Title>
          </div>
        </div>
        {history.length ? (
          <div className="history-list">
            {history.map((item) => (
              <motion.div key={item.id} whileHover={{ y: -2 }} className="history-row">
                <div>
                  <Text className="history-date">{new Date(item.created_at).toLocaleString()}</Text>
                  <Paragraph className="history-explanation">{item.explanation}</Paragraph>
                  {typeof item.key_metrics.model_name === "string" ? (
                    <Text className="muted">{modelLabel(item.key_metrics.model_name)}</Text>
                  ) : null}
                </div>
                <div className="history-metrics">
                  <strong>{item.risk_percent}%</strong>
                  <Tag color={item.risk_band === "high" ? "red" : item.risk_band === "moderate" ? "gold" : "cyan"}>
                    {item.risk_band.toUpperCase()}
                  </Tag>
                  <Text className="muted">Confidence {item.prediction_confidence}%</Text>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <Alert type="info" showIcon message="No history yet" description="Saved runs will appear here after the first prediction." />
        )}
      </Card>
    </motion.div>
  );

  if (bootLoading) {
    return (
      <>
        <BackgroundLayer />
        <Layout className="auth-shell">
          <Content className="boot-loader">
            <Card className="boot-card" bordered={false}>
              <Skeleton active paragraph={{ rows: 6 }} />
            </Card>
          </Content>
        </Layout>
      </>
    );
  }

  if (!profile) {
    return renderAuth();
  }

  return (
    <>
      <BackgroundLayer />
      <Layout className="app-shell">
        <Header className="app-header">
          <div className="header-brand">
          <Text className="eyebrow">Diploma Research Interface</Text>
          <Title level={3} className="header-title">Type 2 Diabetes Risk Estimation</Title>
          <Text className="muted header-subtitle">
            Structured input, PDF-based laboratory extraction, comparative model selection, and saved experimental runs.
          </Text>
          </div>

          <Space size="middle" className="header-actions">
            <div className="user-chip">
            <Avatar src={profile.user.avatar_url || undefined} size={42}>
              {(profile.user.full_name || profile.user.email).slice(0, 1).toUpperCase()}
            </Avatar>
            <div className="user-chip-text">
              <strong>{profile.user.full_name || "Research user"}</strong>
              <Text className="muted">{profile.user.email}</Text>
            </div>
          </div>
            <Button type="primary" className="signout-button" icon={<LogoutOutlined />} onClick={signOut}>
              Sign out
            </Button>
          </Space>
        </Header>

        <Content className="app-content">
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as AppTab)}
          items={[
            { key: "overview", label: "Overview", children: renderOverview() },
            { key: "assessment", label: "Input Data", children: renderAssessment() },
            { key: "prediction", label: "Prediction", children: renderPrediction() },
            { key: "recommendations", label: "Recommendations", children: renderRecommendations() },
            { key: "history", label: "History", children: renderHistory() },
            ...(profile.user.is_admin ? [{ key: "admin", label: "Admin", children: renderAdmin() }] : []),
          ]}
          className="app-tabs"
        />
        </Content>

        {renderLoadingOverlay()}
      </Layout>
    </>
  );
}

export default App;
