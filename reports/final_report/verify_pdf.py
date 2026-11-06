"""Check page boundaries and render every page of the compiled report for review."""
from __future__ import annotations

import json
import re
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "output/pdf/final_report.pdf"
RENDER = ROOT / "tmp/pdfs/final_report"


def main():
    RENDER.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    references, appendix, diagnostics = None, None, None
    details = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.lstrip().upper().startswith("REFERENCES"):
            references = i + 1
        if "APPENDIX A" in text.upper() or "APPENDIX\nA" in text.upper() or "TEAM WORKLOAD AND REPRODUCTION NOTES" in text.upper():
            appendix = i + 1
        if "SUPPLEMENTARY VALIDATION DIAGNOSTICS" in text.upper():
            diagnostics = i + 1
        assert "\ufffd" not in text, f"Replacement character on page {i+1}"
        assert "??" not in text, f"Unresolved reference on page {i+1}"
        outside = []
        for block in page.get_text("dict")["blocks"]:
            x0, y0, x1, y1 = block["bbox"]
            if min(x0, y0) < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                outside.append(block["bbox"])
        assert not outside, f"Content outside page {i+1}: {outside}"
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False).save(RENDER / f"page-{i+1:02d}.png")
        details.append({"page": i+1, "words": len(text.split()), "first_text": text[:100].replace("\n", " ")})
    assert references is not None, "Separate references page not found"
    assert appendix is not None, "Workload appendix not found"
    assert diagnostics is not None, "Supplementary diagnostics appendix not found"
    main_pages = references - 1
    assert main_pages <= 8, f"Main text exceeds limit: {main_pages} pages"
    manuscript = (ROOT / "reports/final_report/final_report.tex").read_text(encoding="utf-8")
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", manuscript) for key in group.split(",")}
    bibkeys = re.findall(r"\\bibitem\{([^}]+)\}", manuscript)
    assert len(bibkeys) == len(set(bibkeys)), "Duplicate bibliography keys"
    assert cited == set(bibkeys), "Missing or unused bibliography entries"
    record = {"pdf": str(PDF.relative_to(ROOT)), "total_pages": len(doc), "main_pages": main_pages,
              "references_start": references, "appendix_start": appendix,
              "diagnostics_start": diagnostics, "appendix_pages": len(doc)-appendix+1,
              "resolved_bibliography_entries": len(bibkeys),
              "page_bounds_and_reference_checks": "passed", "pages": details}
    (ROOT / "reports/final_report/pdf_verification.json").write_text(json.dumps(record,indent=2) + "\n",encoding="utf-8")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
