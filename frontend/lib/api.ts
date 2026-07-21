import axios from "axios";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: API_BASE,
});

export interface Project {
  id: number;
  name: string;
  description?: string;
  created_at: string;
}

export interface Dataset {
  id: number;
  filename: string;
  n_rows: number;
  n_cols: number;
  target_column?: string;
  date_column?: string;
  version: number;
  uploaded_at: string;
}

export interface DatasetProfile {
  n_rows: number;
  n_cols: number;
  columns: string[];
  dtypes: Record<string, string>;
  null_counts: Record<string, number>;
  duplicate_rows: number;
  numeric_summary: Record<string, Record<string, number>>;
  detected_date_columns: string[];
  suggested_target: string | null;
  health_report?: any;
}

export interface Recommendation {
  workflow_type: string;
  reason: string;
  confidence: number;
  candidate_models: string[];
  suggested_metric: string;
}

export interface Experiment {
  id: number;
  dataset_id: number;
  business_objective?: string;
  workflow_type: string;
  status: string;
  best_model_name?: string;
  metrics_json?: any;
  recommendation_reason?: string;
  confidence?: number;
  error?: string;
  model_version?: number;
  model_status?: string;
  pipeline_logs?: string;
  pipeline_progress?: string;
  leaderboard_json?: any;
  model_path?: string;
  created_at: string;
  completed_at?: string;
}
