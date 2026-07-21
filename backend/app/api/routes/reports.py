import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from fpdf import FPDF

from app.core.database import get_db
from app.core.config import REPORTS_DIR
from app.models.models import Experiment, Project
from app.api.deps import get_owned_project

router = APIRouter(prefix="/api/projects/{project_id}/experiments/{experiment_id}", tags=["reports"])


def _safe_str(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).encode("latin-1", errors="replace").decode("latin-1")


def _get_experiment(experiment_id: int, project: Project, db: Session) -> Experiment:
    exp = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.project_id == project.id)
        .first()
    )
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.get("/shap-plot")
def get_shap_plot(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = _get_experiment(experiment_id, project, db)
    if not exp.shap_plot_path or not os.path.exists(exp.shap_plot_path):
        raise HTTPException(status_code=404, detail="No SHAP plot available for this experiment")
    return FileResponse(exp.shap_plot_path, media_type="image/png")


@router.get("/report")
def get_report(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment is not completed yet")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AutoML platform - experiment report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Project: {_safe_str(project.name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Experiment ID: {exp.id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Workflow type: {_safe_str(exp.workflow_type)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Best model: {_safe_str(exp.best_model_name)}", new_x="LMARGIN", new_y="NEXT")
    if exp.recommendation_reason:
        pdf.multi_cell(0, 8, f"Recommendation reason: {_safe_str(exp.recommendation_reason)}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    best_metrics = (exp.metrics_json or {}).get("best_metrics") or (exp.metrics_json or {}).get("metrics") or {}
    if isinstance(best_metrics, dict):
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                pdf.cell(0, 8, f"{_safe_str(k)}: {round(v, 4)}", new_x="LMARGIN", new_y="NEXT")

    # Output Business Insights
    insights = (exp.metrics_json or {}).get("business_insights", [])
    if insights:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Business & Analytical Insights", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for insight in insights:
            pdf.multi_cell(0, 6, f"- {_safe_str(insight)}", new_x="LMARGIN", new_y="NEXT")

    if exp.shap_plot_path and os.path.exists(exp.shap_plot_path):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        title = "Model Visualizations" if exp.workflow_type == "unsupervised" else "SHAP feature importance"
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.image(exp.shap_plot_path, w=170)

    out_path = os.path.join(REPORTS_DIR, f"report_experiment_{exp.id}.pdf")
    pdf.output(out_path)

    return FileResponse(out_path, media_type="application/pdf", filename=f"report_experiment_{exp.id}.pdf")


@router.get("/report-html")
def get_report_html(
    experiment_id: int,
    project: Project = Depends(get_owned_project),
    db: Session = Depends(get_db),
):
    exp = _get_experiment(experiment_id, project, db)
    if exp.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment is not completed yet")

    best_metrics = (exp.metrics_json or {}).get("best_metrics") or (exp.metrics_json or {}).get("metrics") or {}
    insights = (exp.metrics_json or {}).get("business_insights", [])
    
    metrics_items = []
    if isinstance(best_metrics, dict):
        for k, v in best_metrics.items():
            val = round(v, 4) if isinstance(v, (int, float)) else v
            metrics_items.append(f"<li><strong>{_safe_str(k)}:</strong> {_safe_str(str(val))}</li>")
    metrics_html = "".join(metrics_items)
    
    insights_html = "".join([f"<li>{_safe_str(insight)}</li>" for insight in insights])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AutoML Report - Experiment #{exp.id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #1e293b; line-height: 1.6; background-color: #f8fafc; }}
        .container {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 40px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #f1f5f9; padding-bottom: 16px; margin-top: 0; font-size: 28px; }}
        h2 {{ color: #0f172a; margin-top: 32px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; font-size: 20px; }}
        .meta-grid {{ display: grid; grid-template-cols: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .meta-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
        .meta-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700; margin: 0 0 4px 0; }}
        .meta-value {{ font-size: 16px; font-weight: 600; color: #0f172a; margin: 0; }}
        ul {{ padding-left: 20px; margin: 0; }}
        li {{ margin-bottom: 8px; }}
        .badge {{ display: inline-block; background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AutoML platform - Experiment Report</h1>
        
        <div class="meta-grid">
            <div class="meta-card">
                <p class="meta-label">Project</p>
                <p class="meta-value">{_safe_str(project.name)}</p>
            </div>
            <div class="meta-card">
                <p class="meta-label">Experiment ID</p>
                <p class="meta-value">#{exp.id}</p>
            </div>
            <div class="meta-card">
                <p class="meta-label">Workflow Type</p>
                <p class="meta-value"><span class="badge">{exp.workflow_type}</span></p>
            </div>
            <div class="meta-card">
                <p class="meta-label">Best Model</p>
                <p class="meta-value">{_safe_str(exp.best_model_name)}</p>
            </div>
        </div>

        {f'<h2>Recommendation Reason</h2><p>{_safe_str(exp.recommendation_reason)}</p>' if exp.recommendation_reason else ''}
        
        <h2>Model Evaluation Metrics</h2>
        <ul>{metrics_html}</ul>
        
        {f'<h2>Business & Analytical Insights</h2><ul>{insights_html}</ul>' if insights else ''}
    </div>
</body>
</html>
"""
    out_path = os.path.join(REPORTS_DIR, f"report_experiment_{exp.id}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return FileResponse(out_path, media_type="text/html", filename=f"report_experiment_{exp.id}.html")
