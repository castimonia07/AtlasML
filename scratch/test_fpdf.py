from fpdf import FPDF

try:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Business & Analytical Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    # Test multi_cell with 0
    pdf.multi_cell(0, 6, "- Test insight text that should wrap cleanly across the page.")
    print("w=0 Success!")
except Exception as e:
    print("w=0 Failed:", e)

try:
    pdf2 = FPDF()
    pdf2.add_page()
    pdf2.set_font("Helvetica", "B", 13)
    pdf2.cell(0, 8, "Business & Analytical Insights", new_x="LMARGIN", new_y="NEXT")
    pdf2.set_font("Helvetica", "", 10)
    # Test multi_cell with 190
    pdf2.multi_cell(190, 6, "- Test insight text that should wrap cleanly across the page.")
    print("w=190 Success!")
except Exception as e:
    print("w=190 Failed:", e)
