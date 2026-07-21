"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import PlotlyChart from "@/components/PlotlyChart";
import {
  api,
  API_BASE,
  Project,
  Dataset,
  DatasetProfile,
  Recommendation,
  Experiment,
} from "@/lib/api";
import ToastContainer, { ToastMessage } from "@/components/Toast";

interface PredictionLog {
  id: number;
  model_name: string;
  input_data: any[];
  prediction: any[];
  timestamp: string;
}

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const trainingSectionRef = useRef<HTMLElement>(null);

  const [project, setProject] = useState<Project | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);

  // Advanced AutoML States
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [workflowType, setWorkflowType] = useState<string>("supervised");
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);
  const [tuningMode, setTuningMode] = useState<"default" | "custom">("default");
  const [customParamsJson, setCustomParamsJson] = useState<string>(
    JSON.stringify(
      {
        random_forest: {
          n_estimators: [50, 100],
          max_depth: [5, 10]
        }
      },
      null,
      2
    )
  );

  // Unsupervised Workflow States
  const [businessObjective, setBusinessObjective] = useState<string>("customer_segmentation");
  const [advancedUnsupervised, setAdvancedUnsupervised] = useState<boolean>(false);
  const [unsupervisedAlgorithm, setUnsupervisedAlgorithm] = useState<string>("kmeans");
  const [unsupNClusters, setUnsupNClusters] = useState<number>(3);
  const [unsupEps, setUnsupEps] = useState<number>(0.5);
  const [unsupMinSamples, setUnsupMinSamples] = useState<number>(5);
  const [unsupContamination, setUnsupContamination] = useState<number>(0.05);
  const [unsupNComponents, setUnsupNComponents] = useState<number>(2);
  const [targetDatasetId, setTargetDatasetId] = useState<string>("");
  const [driftResult, setDriftResult] = useState<any>(null);
  const [checkingDrift, setCheckingDrift] = useState<boolean>(false);

  // Time Series Workflow States
  const [tsObjective, setTsObjective] = useState<string>("sales_forecasting");
  const [advancedTs, setAdvancedTs] = useState<boolean>(false);
  const [tsAlgorithm, setTsAlgorithm] = useState<string>("arima");
  const [tsP, setTsP] = useState<number>(1);
  const [tsD, setTsD] = useState<number>(1);
  const [tsQ, setTsQ] = useState<number>(1);
  const [tsPSeasonal, setTsPSeasonal] = useState<number>(0);
  const [tsDSeasonal, setTsDSeasonal] = useState<number>(1);
  const [tsQSeasonal, setTsQSeasonal] = useState<number>(0);
  const [tsSSeasonal, setTsSSeasonal] = useState<number>(7);
  const [tsChangepointScale, setTsChangepointScale] = useState<number>(0.05);
  const [monitoringDatasetId, setMonitoringDatasetId] = useState<string>("");
  const [monitoringResult, setMonitoringResult] = useState<any>(null);
  const [checkingMonitoring, setCheckingMonitoring] = useState<boolean>(false);
  
  // Predict States
  const [predictionFormData, setPredictionFormData] = useState<Record<string, string>>({});
  const [predictions, setPredictions] = useState<any>(null);
  const [predictLogs, setPredictLogs] = useState<PredictionLog[]>([]);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Zoom Forecast state
  const [forecastLimit, setForecastLimit] = useState<number>(0); // 0 means show all

  // Comparison State
  const [selectedForComparison, setSelectedForComparison] = useState<number[]>([]);
  const [activeTab, setActiveTab] = useState<"workspace" | "compare">("workspace");

  // Toast State
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  function addToast(text: string, type: "success" | "error" | "info" = "success") {
    const id = Math.random().toString();
    setToasts((prev) => [...prev, { id, type, text }]);
  }

  function removeToast(id: string) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  // Action / Confirmation States
  const [deletingDataset, setDeletingDataset] = useState<Dataset | null>(null);
  const [deletingExperiment, setDeletingExperiment] = useState<Experiment | null>(null);

  async function confirmDeleteDataset() {
    if (!deletingDataset) return;
    try {
      await api.delete(`/api/projects/${id}/datasets/${deletingDataset.id}`);
      addToast("Dataset removed successfully!", "success");
      const dRes = await api.get(`/api/projects/${id}/datasets`);
      setDatasets(dRes.data);
      if (selectedDataset?.id === deletingDataset.id) {
        if (dRes.data.length > 0) {
          selectDataset(dRes.data[0]);
        } else {
          setSelectedDataset(null);
          setProfile(null);
          setRecommendation(null);
        }
      }
    } catch (err: any) {
      addToast("Failed to remove dataset: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setDeletingDataset(null);
    }
  }

  async function stopTraining() {
    if (!activeExperiment) return;
    try {
      await api.post(`/api/projects/${id}/experiments/${activeExperiment.id}/stop`);
      addToast("Training cancellation requested.", "info");
      const res = await api.get(`/api/projects/${id}/experiments/${activeExperiment.id}`);
      setActiveExperiment(res.data);
      loadExperiments();
    } catch (err: any) {
      addToast("Failed to stop training: " + (err.response?.data?.detail || err.message), "error");
    }
  }

  async function confirmDeleteExperiment() {
    if (!deletingExperiment) return;
    try {
      await api.delete(`/api/projects/${id}/experiments/${deletingExperiment.id}`);
      addToast("Model deleted successfully!", "success");
      if (activeExperiment?.id === deletingExperiment.id) {
        setActiveExperiment(null);
      }
      loadExperiments();
    } catch (err: any) {
      addToast("Failed to delete model: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setDeletingExperiment(null);
    }
  }

  function downloadModel(exp: Experiment) {
    if (!exp.model_path) {
      addToast("Model artifact is not ready yet.", "error");
      return;
    }
    const url = `${API_BASE}/api/projects/${id}/experiments/${exp.id}/download`;
    window.open(url, "_blank");
  }

  function downloadReportPdf(exp: Experiment) {
    const url = `${API_BASE}/api/projects/${id}/experiments/${exp.id}/report`;
    window.open(url, "_blank");
  }

  function downloadReportHtml(exp: Experiment) {
    const url = `${API_BASE}/api/projects/${id}/experiments/${exp.id}/report-html`;
    window.open(url, "_blank");
  }

  useEffect(() => {
    loadAll();
  }, [id]);

  useEffect(() => {
    if (!activeExperiment) return;
    if (["completed", "failed"].includes(activeExperiment.status)) return;
    const interval = setInterval(async () => {
      const res = await api.get(`/api/projects/${id}/experiments/${activeExperiment.id}`);
      setActiveExperiment(res.data);
      if (["completed", "failed"].includes(res.data.status)) {
        clearInterval(interval);
        loadExperiments();
        loadPredictionHistory(activeExperiment.id);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [activeExperiment?.id, activeExperiment?.status]);

  async function loadAll() {
    const [pRes, dRes, eRes] = await Promise.all([
      api.get(`/api/projects/${id}`),
      api.get(`/api/projects/${id}/datasets`),
      api.get(`/api/projects/${id}/experiments`),
    ]);
    setProject(pRes.data);
    setEditName(pRes.data.name);
    setEditDesc(pRes.data.description || "");
    setDatasets(dRes.data);
    setExperiments(eRes.data);
    if (dRes.data.length > 0) selectDataset(dRes.data[0]);
  }

  async function updateProject(e: React.FormEvent) {
    e.preventDefault();
    const res = await api.put(`/api/projects/${id}`, {
      name: editName,
      description: editDesc,
    });
    setProject(res.data);
    setIsEditing(false);
    if (selectedDataset) {
      const recRes = await api.get(`/api/projects/${id}/datasets/${selectedDataset.id}/recommend`);
      setRecommendation(recRes.data);
      setWorkflowType(recRes.data.workflow_type);
    }
  }

  async function loadExperiments() {
    const res = await api.get(`/api/projects/${id}/experiments`);
    setExperiments(res.data);
  }

  async function selectDataset(ds: Dataset) {
    setSelectedDataset(ds);
    setRecommendation(null);
    const res = await api.get(`/api/projects/${id}/datasets/${ds.id}/profile`);
    setProfile(res.data);
    if (!res.data.suggested_target) {
      setWorkflowType("unsupervised");
    } else if (res.data.detected_date_columns && res.data.detected_date_columns.length > 0) {
      setWorkflowType("time_series");
    } else {
      setWorkflowType("supervised");
    }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await api.post(`/api/projects/${id}/datasets`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const newDatasets = await api.get(`/api/projects/${id}/datasets`);
      setDatasets(newDatasets.data);
      selectDataset(res.data);
      if (fileInput.current) fileInput.current.value = "";
    } finally {
      setUploading(false);
    }
  }

  async function updateColumns(target: string, date: string) {
    if (!selectedDataset) return;
    const res = await api.post(`/api/projects/${id}/datasets/${selectedDataset.id}/target`, {
      target_column: target || null,
      date_column: date || null,
    });
    selectDataset(res.data);
  }

  async function getRecommendation() {
    if (!selectedDataset) return;
    const res = await api.get(`/api/projects/${id}/datasets/${selectedDataset.id}/recommend`);
    setRecommendation(res.data);
    setWorkflowType(res.data.workflow_type);
  }

  async function train() {
    if (!selectedDataset) return;
    setTraining(true);
    try {
      let customParams = null;
      if (workflowType === "supervised" && hyperparameterTuning && tuningMode === "custom") {
        try {
          customParams = JSON.parse(customParamsJson);
        } catch (err: any) {
          addToast("Invalid Custom Parameter Grid JSON format: " + err.message, "error");
          setTraining(false);
          return;
        }
      } else if (workflowType === "unsupervised" && advancedUnsupervised) {
        customParams = {
          algorithm: unsupervisedAlgorithm,
          n_clusters: unsupNClusters,
          eps: unsupEps,
          min_samples: unsupMinSamples,
          contamination: unsupContamination,
          n_components: unsupNComponents
        };
      }

      const res = await api.post(`/api/projects/${id}/experiments`, {
        dataset_id: selectedDataset.id,
        workflow_type: workflowType,
        hyperparameter_tuning: (workflowType === "supervised" || workflowType === "time_series") ? hyperparameterTuning : false,
        custom_hyperparameters: customParams,
        business_objective: workflowType === "unsupervised" ? businessObjective : (workflowType === "time_series" ? tsObjective : null),
      });
      setActiveExperiment(res.data);
      loadExperiments();
    } finally {
      setTraining(false);
    }
  }

  async function checkDrift() {
    if (!activeExperiment || !targetDatasetId) return;
    setCheckingDrift(true);
    setDriftResult(null);
    try {
      const res = await api.post(`/api/projects/${id}/experiments/${activeExperiment.id}/drift`, {
        target_dataset_id: parseInt(targetDatasetId)
      });
      setDriftResult(res.data);
    } catch (err: any) {
      addToast("Failed to compute drift: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setCheckingDrift(false);
    }
  }

  async function checkForecastMonitoring() {
    if (!activeExperiment || !monitoringDatasetId) return;
    setCheckingMonitoring(true);
    setMonitoringResult(null);
    try {
      const res = await api.post(`/api/projects/${id}/experiments/${activeExperiment.id}/forecast-monitoring`, {
        target_dataset_id: parseInt(monitoringDatasetId)
      });
      setMonitoringResult(res.data);
    } catch (err: any) {
      addToast("Failed to compute forecast monitoring: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setCheckingMonitoring(false);
    }
  }

  async function handleFormPredict(e: React.FormEvent) {
    e.preventDefault();
    if (!activeExperiment || !profile) return;
    setPredictError(null);
    try {
      const record: Record<string, any> = {};
      profile.columns.forEach((col) => {
        if (col !== selectedDataset?.target_column && col !== selectedDataset?.date_column) {
          const val = predictionFormData[col] || "";
          const dtype = profile.dtypes[col] || "";
          if (dtype.includes("int") || dtype.includes("float")) {
            record[col] = parseFloat(val) || 0;
          } else {
            record[col] = val;
          }
        }
      });
      
      const res = await api.post(
        `/api/projects/${id}/experiments/${activeExperiment.id}/predict`,
        { records: [record] }
      );
      setPredictions(res.data);
      loadPredictionHistory(activeExperiment.id);
    } catch (err: any) {
      setPredictError(err.message || "Prediction request failed.");
    }
  }

  async function loadPredictionHistory(expId: number) {
    try {
      const res = await api.get(`/api/projects/${id}/experiments/${expId}/predict/history`);
      setPredictLogs(res.data);
    } catch (err) {
      console.error(err);
    }
  }

  function toggleComparison(expId: number) {
    setSelectedForComparison((prev) =>
      prev.includes(expId) ? prev.filter((item) => item !== expId) : [...prev, expId]
    );
  }

  function handleSelectActiveExperiment(exp: Experiment) {
    setActiveExperiment(exp);
    loadPredictionHistory(exp.id);
    
    // Reset prediction form data with empty inputs
    if (profile && profile.columns) {
      const initialForm: Record<string, string> = {};
      profile.columns.forEach((c) => {
        if (c !== selectedDataset?.target_column && c !== selectedDataset?.date_column) {
          initialForm[c] = "";
        }
      });
      setPredictionFormData(initialForm);
    }

    // Scroll to training / active experiment section
    setTimeout(() => {
      trainingSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }

  if (!project) {
    return (
      <div>
        <Navbar />
        <p className="max-w-5xl mx-auto px-6 py-10 text-sm text-ink/60">loading project…</p>
      </div>
    );
  }

  const comparedList = experiments.filter((e) => selectedForComparison.includes(e.id));

  return (
    <div>
      <Navbar />
      <main className="max-w-5xl mx-auto px-6 py-10 space-y-10">
        
        {/* Navigation Tabs */}
        <div className="border-b border-line flex gap-6 text-sm font-display mb-6">
          <button
            onClick={() => setActiveTab("workspace")}
            className={`py-2 px-1 border-b-2 transition-all ${
              activeTab === "workspace" ? "border-accent text-accent font-semibold" : "border-transparent text-ink/60"
            }`}
          >
            Project Workspace
          </button>
          <button
            onClick={() => setActiveTab("compare")}
            className={`py-2 px-1 border-b-2 transition-all ${
              activeTab === "compare" ? "border-accent text-accent font-semibold" : "border-transparent text-ink/60"
            }`}
          >
            Experiment Comparison ({selectedForComparison.length})
          </button>
        </div>

        {activeTab === "workspace" ? (
          <>
            {/* Project Header Widget */}
            <div className="card p-6">
              {isEditing ? (
                <form onSubmit={updateProject} className="space-y-4">
                  <div>
                    <label className="label-eyebrow block mb-1">Project Name</label>
                    <input
                      className="input max-w-md"
                      required
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label-eyebrow block mb-1">Problem Statement / Description</label>
                    <textarea
                      className="input w-full min-h-[100px]"
                      value={editDesc}
                      onChange={(e) => setEditDesc(e.target.value)}
                      placeholder="Describe your problem (e.g. Forecast sales for next week, Classify customer churn...)"
                    />
                  </div>
                  <div className="flex gap-2">
                    <button type="submit" className="btn-primary">Save Statement</button>
                    <button type="button" className="btn-secondary" onClick={() => setIsEditing(false)}>Cancel</button>
                  </div>
                </form>
              ) : (
                <div className="flex justify-between items-start gap-4">
                  <div>
                    <p className="label-eyebrow mb-2">project #{project.id}</p>
                    <h1 className="font-display text-2xl">{project.name}</h1>
                    {project.description ? (
                      <p className="text-sm text-ink/70 mt-2 bg-paper/50 p-3 rounded border border-line whitespace-pre-wrap">
                        {project.description}
                      </p>
                    ) : (
                      <p className="text-sm text-ink/40 mt-2 italic">No problem statement described yet.</p>
                    )}
                  </div>
                  <button className="btn-secondary shrink-0" onClick={() => {
                    setEditName(project.name);
                    setEditDesc(project.description || "");
                    setIsEditing(true);
                  }}>
                    Edit Problem Statement
                  </button>
                </div>
              )}
            </div>

            {/* Step 1: Datasets (with metadata versioning) */}
            <section className="card p-6">
              <p className="label-eyebrow mb-3">step 1 — dataset</p>
              <form onSubmit={handleUpload} className="flex flex-col sm:flex-row gap-3 mb-5">
                <input ref={fileInput} type="file" accept=".csv" className="input" />
                <button className="btn-primary shrink-0" disabled={uploading}>
                  {uploading ? "uploading..." : "upload csv"}
                </button>
              </form>
              {datasets.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded border text-xs font-display transition-all ${
                        selectedDataset?.id === ds.id
                          ? "border-accent text-accent bg-accentSoft"
                          : "border-line text-ink/70 hover:border-ink/20"
                      }`}
                    >
                      <button
                        onClick={() => selectDataset(ds)}
                        className="font-medium"
                      >
                        {ds.filename} (v{ds.version || 1})
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeletingDataset(ds);
                        }}
                        className="text-ink/30 hover:text-red-500 text-[10px] ml-1.5 font-bold"
                        title="Remove dataset"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Step 2: Health Report Dashboard */}
            {profile && selectedDataset && (
              <section className="card p-6 space-y-6">
                <p className="label-eyebrow">step 2 — dataset profiling &amp; health report</p>
                
                {profile.health_report && (
                  <div className="grid md:grid-cols-3 gap-4 border border-line rounded p-4 bg-paper/50">
                    <div className="flex flex-col justify-center items-center p-3 border-r border-line md:border-r-1 md:border-b-0 border-b">
                      <p className="label-eyebrow mb-1">Data Quality Score</p>
                      <span className={`text-4xl font-display font-bold ${
                        profile.health_report.quality_score >= 85 ? 'text-green-600' :
                        profile.health_report.quality_score >= 70 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {profile.health_report.quality_score}%
                      </span>
                    </div>
                    <div className="col-span-2 space-y-2">
                      <p className="label-eyebrow">Quality Diagnostics</p>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Duplicates: <span className="font-semibold">{profile.health_report.duplicate_percentage}%</span></div>
                        <div>Missing Cells: <span className="font-semibold">{profile.health_report.missing_percentage}%</span></div>
                      </div>
                      {profile.health_report.warnings && profile.health_report.warnings.length > 0 && (
                        <div className="mt-3 text-xs space-y-1">
                          <p className="font-bold text-red-600">Warnings:</p>
                          <ul className="list-disc list-inside space-y-0.5 text-ink/80">
                            {profile.health_report.warnings.map((w: string, idx: number) => (
                              <li key={idx}>{w}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {profile.health_report.recommendations && profile.health_report.recommendations.length > 0 && (
                        <div className="mt-2 text-xs space-y-1">
                          <p className="font-bold text-accent">Actionable Advice:</p>
                          <ul className="list-disc list-inside space-y-0.5 text-ink/80">
                            {profile.health_report.recommendations.map((r: string, idx: number) => (
                              <li key={idx}>{r}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-sm">
                  <Stat label="rows" value={profile.n_rows} />
                  <Stat label="columns" value={profile.n_cols} />
                  <Stat label="feature count" value={profile.columns.length} />
                  <Stat label="missing values" value={Object.values(profile.null_counts || {}).reduce((a: number, b: any) => a + (typeof b === "number" ? b : 0), 0)} />
                  <Stat label="dataset size" value={`${((profile.n_rows * profile.n_cols * 12) / (1024 * 1024)).toFixed(2)} MB`} />
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left label-eyebrow">
                        <th className="py-2 pr-4">column</th>
                        <th className="py-2 pr-4">dtype</th>
                        <th className="py-2 pr-4">nulls</th>
                      </tr>
                    </thead>
                    <tbody>
                      {profile.columns.map((c) => (
                        <tr key={c} className="border-t border-line">
                          <td className="py-1.5 pr-4 flex items-center gap-1">
                            {c}
                            {c === selectedDataset.target_column && <span className="bg-accentSoft text-accent text-[9px] px-1 rounded">Target</span>}
                            {c === selectedDataset.date_column && <span className="bg-blue-100 text-blue-800 text-[9px] px-1 rounded">Date Index</span>}
                          </td>
                          <td className="py-1.5 pr-4 text-ink/60">{profile.dtypes[c]}</td>
                          <td className="py-1.5 pr-4 text-ink/60">{profile.null_counts[c]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <label className="label-eyebrow block mb-1">target column (auto-suggested)</label>
                    <select
                      className="input"
                      value={selectedDataset.target_column || ""}
                      onChange={(e) => updateColumns(e.target.value, selectedDataset.date_column || "")}
                    >
                      <option value="">none (unsupervised)</option>
                      {profile.columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label-eyebrow block mb-1">date column</label>
                    <select
                      className="input"
                      value={selectedDataset.date_column || ""}
                      onChange={(e) => updateColumns(selectedDataset.target_column || "", e.target.value)}
                    >
                      <option value="">none</option>
                      {profile.columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </section>
            )}

            {/* Step 3: Intelligent Recommendations */}
            {selectedDataset && (
              <section className="card p-6">
                <p className="label-eyebrow mb-3">step 3 — workflow recommendation</p>
                <button className="btn-secondary mb-4" onClick={getRecommendation}>
                  analyze &amp; recommend
                </button>
                {recommendation && (
                  <div className="bg-accentSoft border border-accent/30 rounded p-5 space-y-3 text-sm">
                    <div className="flex justify-between items-center">
                      <p className="font-display text-base text-accent font-semibold">
                        Suggested: {recommendation.workflow_type.replace("_", " ").toUpperCase()}
                      </p>
                      <span className="bg-accent text-paper text-xs px-2.5 py-1 rounded-full font-semibold">
                        Confidence: {Math.round(recommendation.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-ink/80">{recommendation.reason}</p>
                    <div className="grid md:grid-cols-2 gap-4 border-t border-accent/20 pt-3 mt-2 text-xs">
                      <div>
                        <span className="font-bold text-accent uppercase tracking-wider block mb-1">Target Evaluation Metric:</span>
                        <code className="bg-paper border border-line px-2 py-0.5 rounded">{recommendation.suggested_metric}</code>
                      </div>
                      <div>
                        <span className="font-bold text-accent uppercase tracking-wider block mb-1">Candidate Models:</span>
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {recommendation.candidate_models.map((m: string) => (
                            <span key={m} className="bg-paper border border-line px-1.5 py-0.5 rounded text-[10px]">{m}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Step 4: Training & Stepper Progress */}
            {selectedDataset && (
              <section ref={trainingSectionRef} className="card p-6 space-y-4">
                <div className="space-y-4">
                  <p className="label-eyebrow">step 4 — pipeline configuration</p>
                  <div className="max-w-md mt-2 space-y-3">
                    <div>
                      <label className="label-eyebrow block mb-1">select workflow type</label>
                      <select
                        className="input"
                        value={workflowType}
                        onChange={(e) => setWorkflowType(e.target.value)}
                      >
                        <option value="supervised">Supervised (Classification/Regression)</option>
                        <option value="unsupervised">Unsupervised (Clustering)</option>
                        <option value="time_series">Time Series (Forecasting)</option>
                      </select>
                    </div>

                    {workflowType === "supervised" && (
                      <div className="space-y-3 border border-line rounded p-4 bg-paper/30">
                        <div className="flex items-center gap-2">
                          <input
                            id="hyperparameter_tuning"
                            type="checkbox"
                            checked={hyperparameterTuning}
                            onChange={(e) => setHyperparameterTuning(e.target.checked)}
                            className="rounded text-accent focus:ring-accent w-4 h-4 cursor-pointer"
                          />
                          <label htmlFor="hyperparameter_tuning" className="text-xs font-display font-medium text-ink cursor-pointer select-none">
                            Enable Hyperparameter Tuning (Slow but Accurate)
                          </label>
                        </div>

                        {hyperparameterTuning && (
                          <div className="pl-6 space-y-3 border-l-2 border-accent/20 pt-1">
                            <div className="flex gap-4 text-xs">
                              <label className="flex items-center gap-1.5 cursor-pointer">
                                <input
                                  type="radio"
                                  name="tuningMode"
                                  value="default"
                                  checked={tuningMode === "default"}
                                  onChange={() => setTuningMode("default")}
                                  className="text-accent focus:ring-accent"
                                />
                                <span>Use Default Grids</span>
                              </label>
                              <label className="flex items-center gap-1.5 cursor-pointer">
                                <input
                                  type="radio"
                                  name="tuningMode"
                                  value="custom"
                                  checked={tuningMode === "custom"}
                                  onChange={() => setTuningMode("custom")}
                                  className="text-accent focus:ring-accent"
                                />
                                <span>Use Custom JSON Grid</span>
                              </label>
                            </div>

                            {tuningMode === "custom" && (
                              <div className="space-y-1.5">
                                <label className="text-[10px] label-eyebrow block text-ink/60">
                                  Custom Parameter Grid (JSON)
                                </label>
                                <textarea
                                  className="input font-mono text-xs h-32 w-full p-2.5"
                                  value={customParamsJson}
                                  onChange={(e) => setCustomParamsJson(e.target.value)}
                                  placeholder={`{\n  "random_forest": {\n    "n_estimators": [50, 100],\n    "max_depth": [5, 10]\n  }\n}`}
                                />
                                <p className="text-[9px] text-ink/40">
                                  Key is model name (e.g. `random_forest`, `xgboost`, `lightgbm`, `catboost`, `logistic_regression`, `linear_regression`). Values must be lists.
                                </p>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {workflowType === "unsupervised" && (
                      <div className="space-y-3 border border-line rounded p-4 bg-paper/30">
                        <div>
                          <label className="label-eyebrow block mb-1">Select Business Need / Objective</label>
                          <select
                            className="input"
                            value={businessObjective}
                            onChange={(e) => setBusinessObjective(e.target.value)}
                          >
                            <option value="customer_segmentation">Customer Segmentation</option>
                            <option value="product_segmentation">Product Segmentation</option>
                            <option value="user_behavior_clustering">User Behavior Clustering</option>
                            <option value="fraud_detection">Fraud Detection</option>
                            <option value="outlier_detection">Outlier Detection</option>
                            <option value="feature_reduction">Feature Reduction / Visualization</option>
                          </select>
                        </div>

                        <div className="flex items-center gap-2 pt-2">
                          <input
                            id="advanced_unsupervised"
                            type="checkbox"
                            checked={advancedUnsupervised}
                            onChange={(e) => setAdvancedUnsupervised(e.target.checked)}
                            className="rounded text-accent focus:ring-accent w-4 h-4 cursor-pointer"
                          />
                          <label htmlFor="advanced_unsupervised" className="text-xs font-display font-medium text-ink cursor-pointer select-none">
                            Enable Advanced Hyperparameter Tuning
                          </label>
                        </div>

                        {advancedUnsupervised && (
                          <div className="pl-6 space-y-3 border-l-2 border-accent/20 pt-1">
                            <div>
                              <label className="label-eyebrow block mb-1">Select Algorithm</label>
                              <select
                                className="input"
                                value={unsupervisedAlgorithm}
                                onChange={(e) => setUnsupervisedAlgorithm(e.target.value)}
                              >
                                <option value="kmeans">KMeans Clustering</option>
                                <option value="agglomerative">Agglomerative Clustering</option>
                                <option value="dbscan">DBSCAN Clustering</option>
                                <option value="isolation_forest">Isolation Forest Anomaly Detection</option>
                                <option value="pca">PCA Dimensionality Reduction</option>
                              </select>
                            </div>

                            {unsupervisedAlgorithm === "kmeans" && (
                              <div>
                                <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Number of Clusters</label>
                                <input
                                  type="number"
                                  min="2"
                                  max="20"
                                  className="input text-xs py-1"
                                  value={unsupNClusters}
                                  onChange={(e) => setUnsupNClusters(parseInt(e.target.value) || 3)}
                                />
                              </div>
                            )}

                            {unsupervisedAlgorithm === "agglomerative" && (
                              <div>
                                <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Number of Clusters</label>
                                <input
                                  type="number"
                                  min="2"
                                  max="20"
                                  className="input text-xs py-1"
                                  value={unsupNClusters}
                                  onChange={(e) => setUnsupNClusters(parseInt(e.target.value) || 3)}
                                />
                              </div>
                            )}

                            {unsupervisedAlgorithm === "dbscan" && (
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Epsilon (eps)</label>
                                  <input
                                    type="number"
                                    step="0.05"
                                    min="0.01"
                                    className="input text-xs py-1"
                                    value={unsupEps}
                                    onChange={(e) => setUnsupEps(parseFloat(e.target.value) || 0.5)}
                                  />
                                </div>
                                <div>
                                  <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Min Samples</label>
                                  <input
                                    type="number"
                                    min="1"
                                    className="input text-xs py-1"
                                    value={unsupMinSamples}
                                    onChange={(e) => setUnsupMinSamples(parseInt(e.target.value) || 5)}
                                  />
                                </div>
                              </div>
                            )}

                            {unsupervisedAlgorithm === "isolation_forest" && (
                              <div>
                                <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Contamination Ratio</label>
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0.001"
                                  max="0.5"
                                  className="input text-xs py-1"
                                  value={unsupContamination}
                                  onChange={(e) => setUnsupContamination(parseFloat(e.target.value) || 0.05)}
                                />
                              </div>
                            )}

                            {unsupervisedAlgorithm === "pca" && (
                              <div>
                                <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Number of Components</label>
                                <input
                                  type="number"
                                  min="1"
                                  className="input text-xs py-1"
                                  value={unsupNComponents}
                                  onChange={(e) => setUnsupNComponents(parseInt(e.target.value) || 2)}
                                />
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {workflowType === "time_series" && (
                      <div className="space-y-3 border border-line rounded p-4 bg-paper/30">
                        <div>
                          <label className="label-eyebrow block mb-1">Select Business Objective</label>
                          <select
                            className="input"
                            value={tsObjective}
                            onChange={(e) => setTsObjective(e.target.value)}
                          >
                            <option value="sales_forecasting">Sales Forecasting</option>
                            <option value="demand_forecasting">Demand Forecasting</option>
                            <option value="revenue_forecasting">Revenue Forecasting</option>
                            <option value="stock_price_trend_analysis">Stock Price Trend Analysis</option>
                            <option value="energy_consumption_forecasting">Energy Consumption Forecasting</option>
                            <option value="website_traffic_forecasting">Website Traffic Forecasting</option>
                            <option value="inventory_forecasting">Inventory Forecasting</option>
                            <option value="weather_forecasting">Weather Forecasting</option>
                            <option value="sensor_forecasting">Sensor Forecasting</option>
                          </select>
                        </div>

                        {!advancedTs && (
                          <div className="flex items-center gap-2 pt-2">
                            <input
                              id="ts_hyperparameter_tuning"
                              type="checkbox"
                              checked={hyperparameterTuning}
                              onChange={(e) => setHyperparameterTuning(e.target.checked)}
                              className="rounded text-accent focus:ring-accent w-4 h-4 cursor-pointer"
                            />
                            <label htmlFor="ts_hyperparameter_tuning" className="text-xs font-display font-medium text-ink cursor-pointer select-none">
                              Enable Hyperparameter Tuning (Slow but Accurate)
                            </label>
                          </div>
                        )}

                        <div className="flex items-center gap-2 pt-2">
                          <input
                            id="advanced_ts"
                            type="checkbox"
                            checked={advancedTs}
                            onChange={(e) => setAdvancedTs(e.target.checked)}
                            className="rounded text-accent focus:ring-accent w-4 h-4 cursor-pointer"
                          />
                          <label htmlFor="advanced_ts" className="text-xs font-display font-medium text-ink cursor-pointer select-none">
                            Enable Advanced Hyperparameter Tuning
                          </label>
                        </div>

                        {advancedTs && (
                          <div className="pl-6 space-y-3 border-l-2 border-accent/20 pt-1">
                            <div>
                              <label className="label-eyebrow block mb-1">Select Algorithm</label>
                              <select
                                className="input"
                                value={tsAlgorithm}
                                onChange={(e) => setTsAlgorithm(e.target.value)}
                              >
                                <option value="arima">ARIMA</option>
                                <option value="sarima">SARIMA</option>
                                <option value="prophet">Prophet (Fourier regression)</option>
                              </select>
                            </div>

                            {tsAlgorithm === "arima" && (
                              <div className="grid grid-cols-3 gap-2">
                                <div>
                                  <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">p (AR order)</label>
                                  <input
                                    type="number"
                                    min="0"
                                    max="5"
                                    className="input text-xs py-1"
                                    value={tsP}
                                    onChange={(e) => setTsP(parseInt(e.target.value) || 0)}
                                  />
                                </div>
                                <div>
                                  <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">d (Diff order)</label>
                                  <input
                                    type="number"
                                    min="0"
                                    max="2"
                                    className="input text-xs py-1"
                                    value={tsD}
                                    onChange={(e) => setTsD(parseInt(e.target.value) || 0)}
                                  />
                                </div>
                                <div>
                                  <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">q (MA order)</label>
                                  <input
                                    type="number"
                                    min="0"
                                    max="5"
                                    className="input text-xs py-1"
                                    value={tsQ}
                                    onChange={(e) => setTsQ(parseInt(e.target.value) || 0)}
                                  />
                                </div>
                              </div>
                            )}

                            {tsAlgorithm === "sarima" && (
                              <div className="space-y-2">
                                <div className="grid grid-cols-3 gap-2">
                                  <div>
                                    <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">p</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="5"
                                      className="input text-xs py-1"
                                      value={tsP}
                                      onChange={(e) => setTsP(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">d</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="2"
                                      className="input text-xs py-1"
                                      value={tsD}
                                      onChange={(e) => setTsD(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">q</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="5"
                                      className="input text-xs py-1"
                                      value={tsQ}
                                      onChange={(e) => setTsQ(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                </div>
                                <div className="grid grid-cols-4 gap-1">
                                  <div>
                                    <label className="text-[9px] label-eyebrow block text-ink/60 mb-0.5">P (Seasonal)</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="2"
                                      className="input text-xs py-1"
                                      value={tsPSeasonal}
                                      onChange={(e) => setTsPSeasonal(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[9px] label-eyebrow block text-ink/60 mb-0.5">D (Seasonal)</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="1"
                                      className="input text-xs py-1"
                                      value={tsDSeasonal}
                                      onChange={(e) => setTsDSeasonal(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[9px] label-eyebrow block text-ink/60 mb-0.5">Q (Seasonal)</label>
                                    <input
                                      type="number"
                                      min="0"
                                      max="2"
                                      className="input text-xs py-1"
                                      value={tsQSeasonal}
                                      onChange={(e) => setTsQSeasonal(parseInt(e.target.value) || 0)}
                                    />
                                  </div>
                                  <div>
                                    <label className="text-[9px] label-eyebrow block text-ink/60 mb-0.5">s (Period)</label>
                                    <input
                                      type="number"
                                      min="1"
                                      className="input text-xs py-1"
                                      value={tsSSeasonal}
                                      onChange={(e) => setTsSSeasonal(parseInt(e.target.value) || 7)}
                                    />
                                  </div>
                                </div>
                              </div>
                            )}

                            {tsAlgorithm === "prophet" && (
                              <div>
                                <label className="text-[10px] label-eyebrow block text-ink/60 mb-0.5">Changepoint Prior Scale</label>
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0.001"
                                  max="1"
                                  className="input text-xs py-1"
                                  value={tsChangepointScale}
                                  onChange={(e) => setTsChangepointScale(parseFloat(e.target.value) || 0.05)}
                                />
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex gap-2">
                  <button className="btn-primary" onClick={train} disabled={training || (activeExperiment && ["pending", "running"].includes(activeExperiment.status))}>
                    {training ? "starting..." : "run pipeline"}
                  </button>
                  {activeExperiment && ["pending", "running"].includes(activeExperiment.status) && (
                    <button
                      onClick={stopTraining}
                      className="bg-red-500 hover:bg-red-600 text-paper text-xs py-2 px-4 rounded font-semibold transition-colors"
                    >
                      Stop Training
                    </button>
                  )}
                </div>

                {activeExperiment && (
                  <div className="mt-5 space-y-4">
                    <div className="flex items-center gap-3">
                      <StatusBadge status={activeExperiment.status} />
                      {activeExperiment.model_version && (
                        <span className="text-xs bg-paper border border-line px-2 py-0.5 rounded font-display">
                          Model v{activeExperiment.model_version} ({activeExperiment.model_status})
                        </span>
                      )}
                    </div>

                    {/* Stepper Progress Visualizer */}
                    {["pending", "running", "completed", "failed", "stopped"].includes(activeExperiment.status) && (
                      <Stepper progress={activeExperiment.pipeline_progress || "pending"} logs={activeExperiment.pipeline_logs || ""} />
                    )}

                    {/* Stdout Live logs console */}
                    {activeExperiment.pipeline_logs && (
                      <div className="space-y-1">
                        <p className="label-eyebrow text-[10px] text-ink/50">Execution Console Logs</p>
                        <pre className="text-[10px] bg-ink text-paper p-4 rounded border border-line font-mono h-40 overflow-y-auto whitespace-pre-wrap select-text">
                          {activeExperiment.pipeline_logs}
                        </pre>
                      </div>
                    )}

                    {activeExperiment.status === "failed" && (
                      <pre className="text-xs text-red-700 mt-2 whitespace-pre-wrap">{activeExperiment.error}</pre>
                    )}

                    {activeExperiment.status === "completed" && (
                      <div className="space-y-6 pt-4 border-t border-line">
                        
                        {/* Business Insights Panel */}
                        {activeExperiment.metrics_json?.business_insights && (
                          <div className="bg-paper border border-line rounded p-5 space-y-2">
                            <h3 className="font-display font-semibold text-accent text-sm uppercase tracking-wider">Business &amp; Analytical Insights</h3>
                            <ul className="list-disc list-inside space-y-1 text-xs text-ink/80">
                              {(activeExperiment.metrics_json.business_insights as string[]).map((insight, idx) => (
                                <li key={idx}>{insight}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Leaderboard panel */}
                        {activeExperiment.leaderboard_json && (
                          <div className="space-y-2">
                            <p className="label-eyebrow">Model Performance Leaderboard</p>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs border border-line">
                                <thead>
                                  <tr className="bg-paper border-b border-line text-left label-eyebrow">
                                    <th className="p-2">Rank</th>
                                    <th className="p-2">Model Candidate</th>
                                    <th className="p-2">Accuracy / R2 / RMSE</th>
                                    <th className="p-2">Precision / Recall / F1 / MAPE</th>
                                    <th className="p-2">Train time</th>
                                    <th className="p-2">Inference</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {(activeExperiment.leaderboard_json as any[]).map((r) => (
                                    <tr key={r.rank} className={`border-b border-line ${r.rank === 1 ? 'bg-accentSoft font-semibold text-accent' : ''}`}>
                                      <td className="p-2">#{r.rank}</td>
                                      <td className="p-2">{r.model_name}</td>
                                      <td className="p-2">
                                        <div className="flex flex-wrap gap-1.5 text-[11px] font-mono">
                                          {r.metrics.accuracy !== undefined && (
                                            <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded">
                                              Acc: <span className="font-bold">{round(r.metrics.accuracy, 4)}</span>
                                            </span>
                                          )}
                                          {r.metrics.r2 !== undefined && (
                                            <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded">
                                              R²: <span className="font-bold">{round(r.metrics.r2, 4)}</span>
                                            </span>
                                          )}
                                          {r.metrics.rmse !== undefined && (
                                            <span className="bg-red-50 text-red-700 border border-red-200 px-1.5 py-0.5 rounded">
                                              RMSE: <span className="font-bold">{round(r.metrics.rmse, 3)}</span>
                                            </span>
                                          )}
                                          {r.metrics.silhouette_score !== undefined && (
                                            <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded">
                                              Sil: <span className="font-bold">{round(r.metrics.silhouette_score, 4)}</span>
                                            </span>
                                          )}
                                        </div>
                                      </td>
                                      <td className="p-2">
                                        <div className="flex flex-wrap gap-1.5 text-[10px] font-mono">
                                          {r.metrics.precision !== undefined && (
                                            <span className="bg-paper border border-line px-1.5 py-0.5 rounded text-ink/85">
                                              P: <span className="font-semibold">{round(r.metrics.precision, 2)}</span> R: <span className="font-semibold">{round(r.metrics.recall, 2)}</span> F1: <span className="font-semibold">{round(r.metrics.f1_weighted, 2)}</span>
                                            </span>
                                          )}
                                          {r.metrics.mape !== undefined && (
                                            <span className="bg-paper border border-line px-1.5 py-0.5 rounded text-ink/85">
                                              MAPE: <span className="font-semibold">{round(r.metrics.mape, 2)}%</span>
                                            </span>
                                          )}
                                        </div>
                                      </td>
                                      <td className="p-2">{r.train_time}s</td>
                                      <td className="p-2">{r.inference_time}s</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Unsupervised Cluster Descriptions panel */}
                        {activeExperiment.workflow_type === "unsupervised" && activeExperiment.metrics_json?.cluster_descriptions && (
                          <div className="space-y-2 card p-4">
                            <p className="label-eyebrow">Cluster Descriptions &amp; Metrics</p>
                            <div className="grid md:grid-cols-2 gap-4 text-xs">
                              {Object.entries(activeExperiment.metrics_json.cluster_descriptions).map(([clusId, desc]: any) => {
                                const stat = activeExperiment.metrics_json.cluster_statistics[clusId];
                                return (
                                  <div key={clusId} className="border border-line rounded p-3 bg-paper/30">
                                    <div className="flex justify-between font-bold text-accent mb-1">
                                      <span>Cluster #{clusId}</span>
                                      <span>{stat?.percentage}% ({stat?.size} samples)</span>
                                    </div>
                                    <p className="text-ink/80 italic mb-2">"{desc}"</p>
                                    <div className="text-[10px] text-ink/60 grid grid-cols-2 gap-1 border-t border-line/40 pt-1.5">
                                      {Object.entries(stat?.means || {}).map(([col, mean]: any) => (
                                        <div key={col} className="truncate">{col}: <span className="font-semibold text-ink">{round(mean, 2)}</span></div>
                                      ))}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Unsupervised Anomaly Summary panel */}
                        {activeExperiment.workflow_type === "unsupervised" && activeExperiment.metrics_json?.anomaly_summary && (
                          <div className="space-y-2 card p-4">
                            <p className="label-eyebrow">Anomaly Detection Summary</p>
                            <div className="border border-line rounded p-4 bg-paper/30 space-y-2 text-xs">
                              <div className="flex justify-between font-bold text-accent">
                                <span>Total Detected Anomalies:</span>
                                <span>{activeExperiment.metrics_json.anomaly_summary.anomaly_count} ({activeExperiment.metrics_json.anomaly_summary.anomaly_percentage}%)</span>
                              </div>
                              {activeExperiment.metrics_json.anomaly_summary.key_observations && activeExperiment.metrics_json.anomaly_summary.key_observations.length > 0 && (
                                <div className="pt-2 border-t border-line/40 space-y-1">
                                  <p className="font-semibold text-ink/75">Key Outlier Observations:</p>
                                  <ul className="list-disc list-inside space-y-1 text-ink/70">
                                    {(activeExperiment.metrics_json.anomaly_summary.key_observations as string[]).map((obs, idx) => (
                                      <li key={idx}>{obs}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Unsupervised PCA Summary panel */}
                        {activeExperiment.workflow_type === "unsupervised" && activeExperiment.metrics_json?.explained_variance_ratio && activeExperiment.metrics_json.explained_variance_ratio.length > 0 && (
                          <div className="space-y-2 card p-4">
                            <p className="label-eyebrow">PCA Explained Variance Breakdown</p>
                            <div className="border border-line rounded p-4 bg-paper/30 space-y-2 text-xs">
                              <p className="font-semibold text-accent">
                                Total Cumulative Variance Explained: {round((activeExperiment.metrics_json.explained_variance_ratio as number[]).reduce((a, b) => a + b, 0) * 100, 2)}%
                              </p>
                              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-line/40">
                                {(activeExperiment.metrics_json.explained_variance_ratio as number[]).map((var_ratio, idx) => (
                                  <div key={idx} className="bg-paper border border-line rounded p-2 text-center">
                                    <span className="text-[10px] label-eyebrow text-ink/50">Component {idx + 1}</span>
                                    <p className="font-display font-bold text-sm text-ink">{round(var_ratio * 100, 2)}%</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Time Series zoom level controls */}
                        {activeExperiment.workflow_type === "time_series" && (
                          <div className="flex justify-end items-center gap-2 text-xs">
                            <span className="label-eyebrow text-[10px]">Forecast View Range:</span>
                            <button
                              onClick={() => setForecastLimit(0)}
                              className={`px-2 py-0.5 rounded border ${forecastLimit === 0 ? 'bg-accent text-paper border-accent' : 'border-line'}`}
                            >
                              Show All
                            </button>
                            <button
                              onClick={() => setForecastLimit(30)}
                              className={`px-2 py-0.5 rounded border ${forecastLimit === 30 ? 'bg-accent text-paper border-accent' : 'border-line'}`}
                            >
                              Last 30 Points
                            </button>
                            <button
                              onClick={() => setForecastLimit(60)}
                              className={`px-2 py-0.5 rounded border ${forecastLimit === 60 ? 'bg-accent text-paper border-accent' : 'border-line'}`}
                            >
                              Last 60 Points
                            </button>
                          </div>
                        )}

                        <ExperimentResults projectId={id} experiment={activeExperiment} forecastLimit={forecastLimit} />

                        {/* Download Model Button */}
                        <div className="flex gap-2">
                          <a
                            href={`${API_BASE}/api/projects/${id}/experiments/${activeExperiment.id}/download`}
                            className="btn-primary inline-flex items-center gap-1"
                            download
                          >
                            Download Model (.joblib)
                          </a>
                        </div>

                        {/* Inference and Prediction Log history */}
                        <div className="grid md:grid-cols-2 gap-6 pt-4 border-t border-line">
                          
                          {/* Run Inference Tester Form */}
                          <div className="space-y-3">
                            <p className="label-eyebrow">Manual Prediction Form</p>
                            <form onSubmit={handleFormPredict} className="space-y-3 border border-line rounded p-4 bg-paper/30">
                              <div className="grid grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-1">
                                {profile.columns.map((col) => {
                                  if (col !== selectedDataset.target_column && col !== selectedDataset.date_column) {
                                    const isNumeric = profile.dtypes[col]?.includes("int") || profile.dtypes[col]?.includes("float");
                                    return (
                                      <div key={col} className="space-y-0.5">
                                        <label className="text-[10px] label-eyebrow block text-ink/70">{col}</label>
                                        <input
                                          type={isNumeric ? "number" : "text"}
                                          step="any"
                                          className="input text-xs py-1 px-2"
                                          value={predictionFormData[col] || ""}
                                          onChange={(e) => setPredictionFormData({
                                            ...predictionFormData,
                                            [col]: e.target.value
                                          })}
                                          placeholder={isNumeric ? "0.0" : "text value"}
                                        />
                                      </div>
                                    );
                                  }
                                  return null;
                                })}
                              </div>
                              {predictError && <p className="text-xs text-red-600 font-semibold">{predictError}</p>}
                              <button type="submit" className="btn-secondary text-xs w-full py-2">Get Predictions</button>
                            </form>

                            {predictions && (
                              <div className="p-3 bg-paper border border-line rounded space-y-2">
                                <p className="label-eyebrow mb-1">Prediction Outputs</p>
                                <div className="text-sm font-semibold text-accent">
                                  Predicted Target: {JSON.stringify(predictions.predictions ?? predictions.cluster)}
                                </div>
                                
                                {/* Probability Meter */}
                                {predictions.probabilities && (
                                  <div className="space-y-1">
                                    <div className="flex justify-between text-[10px] label-eyebrow text-ink/60">
                                      <span>Probability / Confidence Meter</span>
                                      <span>{round(predictions.probabilities[0] * 100, 1)}%</span>
                                    </div>
                                    <div className="w-full bg-line rounded-full h-2">
                                      <div
                                        className="bg-accent h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${predictions.probabilities[0] * 100}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Prediction history */}
                          <div className="space-y-2">
                            <p className="label-eyebrow">Predictions Logs</p>
                            <div className="border border-line rounded divide-y divide-line max-h-[300px] overflow-y-auto bg-paper/50">
                              {predictLogs.length === 0 ? (
                                <p className="text-xs text-ink/50 p-4 text-center">No prediction calls recorded yet.</p>
                              ) : (
                                predictLogs.map((log) => (
                                  <div key={log.id} className="p-3 text-xs space-y-1">
                                    <div className="flex justify-between text-[10px] text-ink/40">
                                      <span>Model: {log.model_name}</span>
                                      <span>{new Date(log.timestamp).toLocaleTimeString()}</span>
                                    </div>
                                    <div>Input: <code className="bg-paper p-0.5 rounded text-[10px]">{JSON.stringify(log.input_data)}</code></div>
                                    <div>Predictions: <span className="font-semibold text-accent">{JSON.stringify(log.prediction)}</span></div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Data Drift & Feature Drift Monitoring Panel */}
                        <div className="pt-6 border-t border-line space-y-4">
                          <h3 className="font-display font-semibold text-accent text-sm uppercase tracking-wider">
                            Production Monitoring (Data &amp; Feature Drift)
                          </h3>
                          <p className="text-xs text-ink/70">
                            Check for statistical drift in feature distributions by comparing this model's training dataset with any other dataset in the project.
                          </p>
                          <div className="flex flex-wrap items-end gap-3 max-w-lg">
                            <div className="flex-1 min-w-[200px]">
                              <label className="text-[10px] label-eyebrow block text-ink/50 mb-1">Select Target Dataset</label>
                              <select
                                className="input py-1.5 text-xs"
                                value={targetDatasetId}
                                onChange={(e) => setTargetDatasetId(e.target.value)}
                              >
                                <option value="">-- select dataset --</option>
                                {datasets
                                  .filter((d) => d.id !== activeExperiment.dataset_id)
                                  .map((d) => (
                                    <option key={d.id} value={d.id}>
                                      {d.filename} (v{d.version})
                                    </option>
                                  ))}
                              </select>
                            </div>
                            <button
                              onClick={checkDrift}
                              disabled={checkingDrift || !targetDatasetId}
                              className="btn-secondary py-1.5 px-4 text-xs font-semibold"
                            >
                              {checkingDrift ? "Calculating Drift..." : "Check for Drift"}
                            </button>
                          </div>

                          {driftResult && (
                            <div className="border border-line rounded p-4 bg-paper/30 space-y-3">
                              <div className="flex justify-between items-center">
                                <span className="label-eyebrow text-xs">Drift Detection Status</span>
                                <span className={`text-xs font-display px-2.5 py-0.5 rounded font-semibold ${
                                  driftResult.drift_detected 
                                    ? "bg-red-100 text-red-800" 
                                    : "bg-green-100 text-green-800"
                                }`}>
                                  {driftResult.drift_detected ? "DRIFT DETECTED" : "NO SIGNIFICANT DRIFT"}
                                </span>
                              </div>
                              <p className="text-xs font-semibold text-ink/80">{driftResult.summary}</p>
                              
                              <div className="overflow-x-auto pt-2">
                                <table className="w-full text-left text-[11px] border-collapse text-ink">
                                  <thead>
                                    <tr className="border-b border-line text-ink/50 uppercase tracking-wider font-semibold label-eyebrow">
                                      <th className="pb-1.5">Feature Column</th>
                                      <th className="pb-1.5">Test Method</th>
                                      <th className="pb-1.5 text-right">P-Value</th>
                                      <th className="pb-1.5 text-right">Drift Status</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(driftResult.features).map(([feat, det]: any) => (
                                      <tr key={feat} className="border-b border-line/45">
                                        <td className="py-2 font-mono font-medium">{feat}</td>
                                        <td className="py-2 text-ink/70">{det.test_method}</td>
                                        <td className="py-2 text-right font-mono">{det.p_value.toFixed(5)}</td>
                                        <td className="py-2 text-right">
                                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                            det.drift_detected 
                                              ? "bg-red-50 text-red-700" 
                                              : "bg-green-50 text-green-700"
                                          }`}>
                                            {det.drift_detected ? "Drifted" : "Stable"}
                                          </span>
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Forecast Error Monitoring Panel */}
                        {activeExperiment.workflow_type === "time_series" && (
                          <div className="pt-6 border-t border-line space-y-4">
                            <h3 className="font-display font-semibold text-accent text-sm uppercase tracking-wider">
                              Forecast Performance &amp; Error Degradation Monitoring
                            </h3>
                            <p className="text-xs text-ink/70">
                              Compare predictions with a newly uploaded dataset containing actual target metrics to calculate forecast error (RMSE, MAE, MAPE) and detect performance degradation.
                            </p>
                            <div className="flex flex-wrap items-end gap-3 max-w-lg">
                              <div className="flex-1 min-w-[200px]">
                                <label className="text-[10px] label-eyebrow block text-ink/50 mb-1">Select Validation Dataset</label>
                                <select
                                  className="input py-1.5 text-xs"
                                  value={monitoringDatasetId}
                                  onChange={(e) => setMonitoringDatasetId(e.target.value)}
                                >
                                  <option value="">-- select dataset --</option>
                                  {datasets
                                    .filter((d) => d.id !== activeExperiment.dataset_id)
                                    .map((d) => (
                                      <option key={d.id} value={d.id}>
                                        {d.filename} (v{d.version})
                                      </option>
                                    ))}
                                </select>
                              </div>
                              <button
                                onClick={checkForecastMonitoring}
                                disabled={checkingMonitoring || !monitoringDatasetId}
                                className="btn-secondary py-1.5 px-4 text-xs font-semibold"
                              >
                                {checkingMonitoring ? "Checking Forecast..." : "Check Forecast Performance"}
                              </button>
                            </div>

                            {monitoringResult && (
                              <div className="border border-line rounded p-4 bg-paper/30 space-y-3">
                                <div className="flex justify-between items-center">
                                  <span className="label-eyebrow text-xs">Model Degradation Status</span>
                                  <span className={`text-xs font-display px-2.5 py-0.5 rounded font-semibold ${
                                    monitoringResult.status === "Degraded" 
                                      ? "bg-red-100 text-red-800" 
                                      : "bg-green-100 text-green-800"
                                  }`}>
                                    {monitoringResult.status.toUpperCase()}
                                  </span>
                                </div>
                                <p className="text-xs font-semibold text-ink/80">{monitoringResult.summary}</p>
                                
                                <div className="grid grid-cols-3 gap-3 pt-2">
                                  <div className="bg-paper border border-line rounded p-2 text-center">
                                    <span className="text-[9px] label-eyebrow text-ink/50">MAPE</span>
                                    <p className="font-display font-bold text-sm text-ink">{monitoringResult.metrics.mape.toFixed(2)}%</p>
                                    <span className="text-[8px] text-ink/40">Baseline: {monitoringResult.baseline_metrics.mape?.toFixed(2)}%</span>
                                  </div>
                                  <div className="bg-paper border border-line rounded p-2 text-center">
                                    <span className="text-[9px] label-eyebrow text-ink/50">RMSE</span>
                                    <p className="font-display font-bold text-sm text-ink">{monitoringResult.metrics.rmse.toFixed(2)}</p>
                                    <span className="text-[8px] text-ink/40">Baseline: {monitoringResult.baseline_metrics.rmse?.toFixed(2)}</span>
                                  </div>
                                  <div className="bg-paper border border-line rounded p-2 text-center">
                                    <span className="text-[9px] label-eyebrow text-ink/50">MAE</span>
                                    <p className="font-display font-bold text-sm text-ink">{monitoringResult.metrics.mae.toFixed(2)}</p>
                                    <span className="text-[8px] text-ink/40">Baseline: {monitoringResult.baseline_metrics.mae?.toFixed(2)}</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </section>
            )}

            {/* Past experiments List (with checkboxes for comparison selection) */}
            {experiments.length > 0 && (
              <section className="space-y-3">
                <p className="label-eyebrow">experiment history &amp; comparison selection</p>
                <div className="space-y-3">
                  {experiments.map((exp) => (
                    <div key={exp.id} className="card p-4 flex items-center justify-between gap-4">
                      <div className="flex items-center gap-3 min-w-0">
                        <input
                          type="checkbox"
                          checked={selectedForComparison.includes(exp.id)}
                          onChange={() => toggleComparison(exp.id)}
                          className="rounded text-accent focus:ring-accent shrink-0"
                        />
                        <div className="min-w-0">
                          <p className="text-sm font-display flex flex-wrap items-center gap-2 break-all">
                            #{exp.id} · {exp.workflow_type.replace("_", " ")} · {exp.best_model_name || "—"}
                            {exp.model_version && <span className="text-[10px] bg-paper border border-line px-1 rounded shrink-0">v{exp.model_version}</span>}
                            {exp.model_status === "production" && <span className="bg-green-100 text-green-800 text-[9px] px-1 rounded font-display font-semibold shrink-0">Production</span>}
                          </p>
                          <p className="text-xs text-ink/50">{new Date(exp.created_at).toLocaleString()}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <StatusBadge status={exp.status} />
                        <button className="label-eyebrow underline text-xs font-semibold" onClick={() => handleSelectActiveExperiment(exp)}>
                          view
                        </button>
                        {exp.status === "completed" && (
                          <>
                            <span className="text-line">|</span>
                            <button
                              onClick={() => downloadModel(exp)}
                              className="text-[10px] text-accent font-semibold uppercase tracking-wider hover:underline"
                              title="Download model (.joblib)"
                            >
                              Download
                            </button>
                            <span className="text-line">|</span>
                            <button
                              onClick={() => downloadReportPdf(exp)}
                              className="text-[10px] text-blue-600 font-semibold uppercase tracking-wider hover:underline"
                              title="Download PDF report"
                            >
                              PDF
                            </button>
                            <span className="text-line">|</span>
                            <button
                              onClick={() => downloadReportHtml(exp)}
                              className="text-[10px] text-green-600 font-semibold uppercase tracking-wider hover:underline"
                              title="Download HTML report"
                            >
                              HTML
                            </button>
                          </>
                        )}
                        <span className="text-line">|</span>
                        <button
                          onClick={() => setDeletingExperiment(exp)}
                          className="p-1 hover:bg-red-50 rounded transition-colors text-red-500 hover:text-red-700"
                          title="Delete model"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            strokeWidth={1.5}
                            stroke="currentColor"
                            className="w-4 h-4"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </>
        ) : (
          /* Side-by-Side Comparison Workspace */
          <div className="space-y-6">
            <h2 className="font-display text-xl">Experiment Comparison Panel</h2>
            {comparedList.length < 2 ? (
              <p className="text-sm text-ink/60 bg-paper border border-line p-6 text-center rounded">
                Please select at least 2 experiments in the checklist on the "Project Workspace" tab to compare their performance.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs border border-line">
                  <thead>
                    <tr className="bg-paper border-b border-line text-left label-eyebrow">
                      <th className="p-3">Metric / Feature</th>
                      {comparedList.map((e) => (
                        <th key={e.id} className="p-3 border-l border-line min-w-[160px]">
                          Experiment #{e.id} {e.model_version ? `(v${e.model_version})` : ""}
                          {e.model_status === "production" && <span className="bg-green-100 text-green-800 text-[9px] px-1.5 ml-2 rounded font-semibold">Prod</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-line bg-paper/30">
                      <td className="p-3 font-semibold">Workflow / Best Model</td>
                      {comparedList.map((e) => (
                        <td key={e.id} className="p-3 border-l border-line font-mono text-[10px]">
                          {e.workflow_type.toUpperCase()} / {e.best_model_name}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-line">
                      <td className="p-3 font-semibold">Primary Metric</td>
                      {comparedList.map((e) => {
                        const m = e.metrics_json?.best_metrics || {};
                        return (
                          <td key={e.id} className="p-3 border-l border-line">
                            <div className="flex flex-wrap gap-1.5 text-[11px] font-mono">
                              {m.accuracy !== undefined && (
                                <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded font-normal">
                                  Acc: <span className="font-bold text-accent">{round(m.accuracy, 4)}</span>
                                </span>
                              )}
                              {m.r2 !== undefined && (
                                <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded font-normal">
                                  R²: <span className="font-bold text-accent">{round(m.r2, 4)}</span>
                                </span>
                              )}
                              {m.rmse !== undefined && (
                                <span className="bg-red-50 text-red-700 border border-red-200 px-1.5 py-0.5 rounded font-normal">
                                  RMSE: <span className="font-bold text-red-800">{round(m.rmse, 3)}</span>
                                </span>
                              )}
                              {m.silhouette_score !== undefined && (
                                <span className="bg-accentSoft/30 text-accent border border-accent/20 px-1.5 py-0.5 rounded font-normal">
                                  Sil: <span className="font-bold text-accent">{round(m.silhouette_score, 4)}</span>
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-line">
                      <td className="p-3 font-semibold">Precision / Recall / F1 / MAPE</td>
                      {comparedList.map((e) => {
                        const m = e.metrics_json?.best_metrics || {};
                        return (
                          <td key={e.id} className="p-3 border-l border-line">
                            <div className="flex flex-wrap gap-1.5 text-[10px] font-mono">
                              {m.f1_weighted !== undefined && (
                                <span className="bg-paper border border-line px-1.5 py-0.5 rounded text-ink/80">
                                  P: <span className="font-semibold">{round(m.precision, 2)}</span> R: <span className="font-semibold">{round(m.recall, 2)}</span> F1: <span className="font-semibold">{round(m.f1_weighted, 2)}</span>
                                </span>
                              )}
                              {m.mape !== undefined && (
                                <span className="bg-paper border border-line px-1.5 py-0.5 rounded text-ink/80">
                                  MAPE: <span className="font-semibold">{round(m.mape, 2)}%</span>
                                </span>
                              )}
                              {m.f1_weighted === undefined && m.mape === undefined && <span>—</span>}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-line">
                      <td className="p-3 font-semibold">Training Time</td>
                      {comparedList.map((e) => {
                        const leader = e.leaderboard_json as any[] | null;
                        const t = leader && leader.length > 0 ? leader[0].train_time : 0.0;
                        return (
                          <td key={e.id} className="p-3 border-l border-line">
                            {t}s
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-line">
                      <td className="p-3 font-semibold">Created Date</td>
                      {comparedList.map((e) => (
                        <td key={e.id} className="p-3 border-l border-line text-ink/60">
                          {new Date(e.created_at).toLocaleString()}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Delete Dataset Confirmation Modal */}
      {deletingDataset && (
        <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-paper border border-line rounded-lg shadow-xl max-w-sm w-full p-6 space-y-4">
            <h3 className="font-display font-semibold text-lg text-red-500">Remove Dataset?</h3>
            <p className="text-xs text-ink/70">
              Are you sure you want to remove <strong>{deletingDataset.filename}</strong>? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setDeletingDataset(null)}
                className="btn-secondary text-xs py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteDataset}
                className="bg-red-500 hover:bg-red-600 text-paper rounded text-xs py-1.5 px-3 font-semibold transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Model Confirmation Modal */}
      {deletingExperiment && (
        <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-paper border border-line rounded-lg shadow-xl max-w-sm w-full p-6 space-y-4">
            <h3 className="font-display font-semibold text-lg text-red-500">Delete Model?</h3>
            <p className="text-xs text-ink/70">
              Are you sure you want to delete Model <strong>#{deletingExperiment.id} ({deletingExperiment.best_model_name})</strong>? Artifacts and training statistics will be removed.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setDeletingExperiment(null)}
                className="btn-secondary text-xs py-1.5 px-3"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteExperiment}
                className="bg-red-500 hover:bg-red-600 text-paper rounded text-xs py-1.5 px-3 font-semibold transition-colors"
              >
                Delete Permanently
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast Messages */}
      <ToastContainer toasts={toasts} onClose={removeToast} />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <p className="label-eyebrow mb-1">{label}</p>
      <p className="font-display text-lg">{value}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: "bg-line text-ink/60",
    running: "bg-accentSoft text-accent animate-pulse",
    completed: "bg-accent text-paper",
    failed: "bg-red-100 text-red-800",
    stopped: "bg-orange-100 text-orange-800",
  };
  return (
    <span className={`text-xs font-display px-2.5 py-1 rounded font-semibold ${colors[status] || ""}`}>
      {status.toUpperCase()}
    </span>
  );
}

function Stepper({ progress, logs = "" }: { progress: string; logs?: string }) {
  const stages = ["pending", "cleaning", "engineering", "training", "evaluation", "shap", "completed"];
  
  const progressMap: Record<string, number> = {
    pending: 0,
    cleaning: 1,
    engineering: 2,
    training: 3,
    evaluation: 4,
    shap: 5,
    completed: 6,
    failed: 3
  };

  const currentIndex = progressMap[progress] || 0;

  // Parse logs to extract durations and current model
  const stageTimes: Record<string, string> = {};
  let currentModel: string | null = null;

  if (logs) {
    const lines = logs.split("\n");
    const stageTimestamps: { stage: string; timestamp: Date }[] = [];
    const logRegex = /^\[(.*?)\]\s+(\w+):\s+(.*)$/;
    
    lines.forEach((line) => {
      const match = line.match(logRegex);
      if (match) {
        const tsStr = match[1];
        const stage = match[2].toLowerCase();
        const msg = match[3];
        
        // Clean dash formatting to slashes for Safari support
        const ts = new Date(tsStr.replace(/-/g, "/"));
        if (!isNaN(ts.getTime())) {
          stageTimestamps.push({ stage, timestamp: ts });
        }
        
        // Extract model being trained
        if (stage === "training" && msg.toLowerCase().includes("fitting")) {
          const modelMatch = msg.match(/fitting\s+(\w+)\s+estimator/i);
          if (modelMatch) {
            currentModel = modelMatch[1].toUpperCase();
          }
        }
      }
    });

    // Calculate elapsed durations
    for (let i = 0; i < stageTimestamps.length - 1; i++) {
      const current = stageTimestamps[i];
      const next = stageTimestamps[i + 1];
      if (current.stage !== next.stage) {
        const diffMs = next.timestamp.getTime() - current.timestamp.getTime();
        stageTimes[current.stage] = `${(diffMs / 1000).toFixed(1)}s`;
      }
    }

    // Last stage duration
    if (stageTimestamps.length > 0) {
      const last = stageTimestamps[stageTimestamps.length - 1];
      if (["completed", "failed", "stopped"].includes(last.stage)) {
        let prevIndex = stageTimestamps.length - 2;
        while (prevIndex >= 0 && stageTimestamps[prevIndex].stage === last.stage) {
          prevIndex--;
        }
        if (prevIndex >= 0) {
          const prev = stageTimestamps[prevIndex];
          const diffMs = last.timestamp.getTime() - prev.timestamp.getTime();
          stageTimes[prev.stage] = `${(diffMs / 1000).toFixed(1)}s`;
        }
      } else {
        // If still running, calculate duration of current stage since it started
        let firstOccur = stageTimestamps.findIndex(t => t.stage === last.stage);
        if (firstOccur !== -1) {
          const diffMs = Date.now() - stageTimestamps[firstOccur].timestamp.getTime();
          if (diffMs > 0 && diffMs < 3600000) {
            stageTimes[last.stage] = `${(diffMs / 1000).toFixed(1)}s...`;
          }
        }
      }
    }
  }

  return (
    <div className="w-full py-4 bg-paper/20 rounded border border-line/40 px-4 my-2">
      <div className="flex items-center justify-between text-center max-w-3xl mx-auto text-[10px] font-display label-eyebrow text-ink/50 uppercase">
        {stages.map((stage, idx) => {
          const isCurrent = idx === currentIndex;
          const isCompleted = idx < currentIndex;
          const duration = stageTimes[stage];
          
          return (
            <div key={stage} className="flex-1 relative flex flex-col items-center">
              {/* Connector line */}
              {idx > 0 && (
                <div className={`absolute top-3 -left-1/2 right-1/2 h-[2px] -z-10 ${
                  idx <= currentIndex ? 'bg-accent' : 'bg-line'
                }`} />
              )}
              
              {/* Step circle */}
              <div className={`w-6 h-6 rounded-full border-2 bg-paper flex items-center justify-center relative z-10 transition-all ${
                isCompleted ? 'bg-accent border-accent text-paper font-bold' :
                isCurrent ? 'border-accent text-accent font-bold scale-110 shadow-sm animate-pulse' : 'border-line text-ink/30'
              }`}>
                {isCompleted ? "✓" : idx + 1}
              </div>
              
              {/* Step text */}
              <span className={`mt-2 block text-[9px] ${isCurrent ? 'text-accent font-semibold' : 'text-ink/60'}`}>
                {stage === "shap" ? "SHAP" : stage}
              </span>
              
              {/* Duration and dynamic tracking metadata */}
              {duration && (
                <span className="text-[8px] font-mono text-ink/40 normal-case mt-0.5">
                  {duration}
                </span>
              )}
              
              {stage === "training" && isCurrent && currentModel && (
                <span className="text-[8px] text-accent/80 font-medium normal-case animate-pulse mt-0.5 truncate max-w-[80px]">
                  ({currentModel})
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function round(val: any, decimals: number): string {
  if (typeof val !== "number") return "—";
  return val.toFixed(decimals);
}

function ExperimentResults({ projectId, experiment, forecastLimit = 0 }: { projectId: string; experiment: Experiment; forecastLimit?: number }) {
  const result = experiment.metrics_json;
  if (!result) return null;

  const reportUrl = `${API_BASE}/api/projects/${projectId}/experiments/${experiment.id}/report`;
  const shapUrl = `${API_BASE}/api/projects/${projectId}/experiments/${experiment.id}/shap-plot`;

  // Filter time series values if forecastLimit is selected
  let actualSeries = result.actual || [];
  let predictedSeries = result.predicted || [];
  if (experiment.workflow_type === "time_series" && forecastLimit > 0) {
    actualSeries = actualSeries.slice(-forecastLimit);
    predictedSeries = predictedSeries.slice(-forecastLimit);
  }

  return (
    <div className="mt-4 space-y-6">
      
      {/* Supervised comparison chart */}
      {experiment.workflow_type === "supervised" && result.metrics && (
        <div className="grid md:grid-cols-2 gap-4">
          <PlotlyChart
            data={[
              {
                type: "bar",
                x: Object.keys(result.metrics),
                y: Object.values(result.metrics).map((m: any) =>
                  m.accuracy ?? m.r2 ?? 0
                ),
                marker: { color: "#3B6D5C" },
              },
            ]}
            layout={{ title: "Model Comparison (Primary Metric)" }}
          />

          {/* Classification Curves: ROC and Precision-Recall */}
          {result.curves && (
            <PlotlyChart
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  x: result.curves.roc_curve?.fpr || [],
                  y: result.curves.roc_curve?.tpr || [],
                  name: `ROC Curve (AUC: ${round(result.curves.roc_curve?.auc, 3)})`,
                  line: { color: "#3B6D5C", width: 2 }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  x: [0, 1],
                  y: [0, 1],
                  name: "Random Baseline",
                  line: { color: "#ccc", dash: "dash" }
                }
              ]}
              layout={{
                title: "ROC Curve (FPR vs TPR)",
                xaxis: { title: "False Positive Rate" },
                yaxis: { title: "True Positive Rate" }
              }}
            />
          )}
        </div>
      )}

      {/* Classification Confusion Matrix rendering */}
      {experiment.workflow_type === "supervised" && result.curves?.confusion_matrix && (
        <div className="card p-4 space-y-2">
          <p className="label-eyebrow">Confusion Matrix Heatmap</p>
          <div className="flex justify-center">
            <div className="border border-line rounded overflow-hidden">
              <table className="text-xs border-collapse">
                <tbody>
                  {(result.curves.confusion_matrix as number[][]).map((row, rIdx) => (
                    <tr key={rIdx} className="border-b border-line">
                      {row.map((val, cIdx) => (
                        <td
                          key={cIdx}
                          className="p-5 text-center font-mono font-bold text-sm border-r border-line"
                          style={{
                            backgroundColor: rIdx === cIdx ? "rgba(59, 109, 92, 0.15)" : "transparent",
                            color: rIdx === cIdx ? "#3B6D5C" : "inherit"
                          }}
                        >
                          <div className="text-[10px] text-ink/40 font-normal mb-1">
                            {rIdx === cIdx ? "True Class" : "Error"}
                          </div>
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Unsupervised Elbow and Silhouette Plot */}
      {experiment.workflow_type === "unsupervised" && result.elbow_curve && (
        <div className="grid md:grid-cols-2 gap-4">
          <PlotlyChart
            data={[
              {
                type: "scatter",
                mode: "lines+markers",
                x: result.elbow_curve.map((p: any) => p.k),
                y: result.elbow_curve.map((p: any) => p.wcss),
                line: { color: "#3B6D5C" }
              }
            ]}
            layout={{
              title: "Elbow Curve (WCSS vs k)",
              xaxis: { title: "Number of Clusters k" },
              yaxis: { title: "WCSS (Inertia)" }
            }}
          />
          <PlotlyChart
            data={[
              {
                type: "bar",
                x: (experiment.leaderboard_json as any[] || []).map((r) => r.model_name.replace("kmeans_k", "k=")),
                y: (experiment.leaderboard_json as any[] || []).map((r) => r.metrics.silhouette_score),
                marker: { color: "#6A5ACD" }
              }
            ]}
            layout={{
              title: "Silhouette Scores Comparison",
              xaxis: { title: "Cluster Size k" },
              yaxis: { title: "Silhouette Score" }
            }}
          />
        </div>
      )}

      {/* PCA Cluster points visualization */}
      {experiment.workflow_type === "unsupervised" && result.pca_points && (
        <PlotlyChart
          data={[
            {
              type: "scattergl",
              mode: "markers",
              x: result.pca_points.map((p: number[]) => p[0]),
              y: result.pca_points.map((p: number[]) => p[1]),
              marker: {
                color: result.cluster_labels,
                colorscale: "Viridis",
                size: 6,
              },
            },
          ]}
          layout={{ title: "PCA Projection clusters view" }}
        />
      )}

      {/* Time Series Forecasting zoom graph */}
      {experiment.workflow_type === "time_series" && result.actual && (
        <PlotlyChart
          data={[
            { type: "scatter", mode: "lines", y: actualSeries, name: "actual", line: { color: "#12151B" } },
            { type: "scatter", mode: "lines", y: predictedSeries, name: "predicted", line: { color: "#3B6D5C" } },
          ]}
          layout={{ title: forecastLimit > 0 ? `Forecast vs actual (Last ${forecastLimit} points)` : "Forecast vs actual" }}
        />
      )}

      {/* Metrics breakdown stats grid */}
      <div className="grid sm:grid-cols-3 gap-4 text-sm">
        {Object.entries(result.best_metrics || result.metrics || {}).map(([k, v]) => (
          <div key={k} className="border border-line rounded p-3 bg-paper/50">
            <p className="label-eyebrow mb-1 text-[10px] text-ink/50 uppercase tracking-wider">{k.replace("_weighted", "")}</p>
            <p className="font-display font-semibold text-base">{typeof v === "number" ? v.toFixed(4) : String(v)}</p>
          </div>
        ))}
      </div>

      {experiment.workflow_type === "supervised" && result.shap_plot_path && (
        <div>
          <p className="label-eyebrow mb-2">shap feature importance</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={shapUrl} alt="SHAP summary plot" className="max-w-full border border-line rounded" />
        </div>
      )}

      <div className="flex gap-2 pt-2">
        <a href={reportUrl} className="btn-secondary inline-block text-xs py-2 px-4 font-semibold" target="_blank" rel="noreferrer">
          Download PDF Report
        </a>
        <a href={`${API_BASE}/api/projects/${projectId}/experiments/${experiment.id}/report-html`} className="btn-secondary inline-block text-xs py-2 px-4 font-semibold" target="_blank" rel="noreferrer">
          Download HTML Report
        </a>
      </div>
    </div>
  );
}
