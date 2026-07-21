import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.database import SessionLocal
from app.models.models import Experiment, Project
from fpdf import FPDF

db = SessionLocal()
try:
    exp = db.query(Experiment).filter(Experiment.id == 7).first()
    project = db.query(Project).filter(Project.id == exp.project_id).first()
    insights = (exp.metrics_json or {}).get("business_insights", [])
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Business & Analytical Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    for i, insight in enumerate(insights):
        print(f"Before insight {i}: x={pdf.get_x()}, y={pdf.get_y()}")
        # Pass new_x and new_y to multi_cell
        pdf.multi_cell(0, 6, f"- {insight}", new_x="LMARGIN", new_y="NEXT")
        print(f"After insight {i}: x={pdf.get_x()}, y={pdf.get_y()}")
        
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    db.close()
