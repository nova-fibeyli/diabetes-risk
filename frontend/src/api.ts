import type {
  AdminOverviewResponse,
  AuthConfigResponse,
  FeatureSchemaResponse,
  HistoryResponse,
  ModelInfoResponse,
  ParsePdfResponse,
  PredictionInput,
  PredictionResponse,
  ProfileResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...init,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return {} as T;
  }
  return response.json() as Promise<T>;
}

export function getAuthConfig() {
  return request<AuthConfigResponse>("/auth/google");
}

export function getProfile() {
  return request<ProfileResponse>("/profile");
}

export function logout() {
  return request<void>("/auth/logout", {
    method: "POST",
  });
}

export function getHistory() {
  return request<HistoryResponse>("/history");
}

export function getAdminOverview() {
  return request<AdminOverviewResponse>("/admin/overview");
}

export function getFeatureSchema() {
  return request<FeatureSchemaResponse>("/feature-schema");
}

export function getModelInfo() {
  return request<ModelInfoResponse>("/model-info");
}

export function predict(input: PredictionInput, modelName?: string) {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, model_name: modelName }),
  });
}

export function parsePdf(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request<ParsePdfResponse>("/parse-pdf", {
    method: "POST",
    body: formData,
  });
}

export function getGoogleLoginUrl() {
  return `${API_BASE}/auth/google/login`;
}
