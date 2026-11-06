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
    fine_tuning, scratch_transformer = None, None
    interpretation = None
    auxiliary_temporal = None
    details = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.lstrip().upper().startswith("REFERENCES"):
            references = i + 1
        if "APPENDIX A" in text.upper() or "APPENDIX\nA" in text.upper() or "TEAM WORKLOAD AND REPRODUCTION NOTES" in text.upper():
            appendix = i + 1
        if "SUPPLEMENTARY VALIDATION DIAGNOSTICS" in text.upper():
            diagnostics = i + 1
        if "TABPFN FINE-TUNING EXPERIMENTS" in text.upper():
            fine_tuning = i + 1
        if "SMALL TRANSFORMERS TRAINED FROM SCRATCH VERSUS TABPFN" in text.upper():
            scratch_transformer = i + 1
        if "INTERPRETING TABPFN'S PREDICTIVE ADVANTAGE" in text.upper() or "INTERPRETING TABPFN’S PREDICTIVE ADVANTAGE" in text.upper():
            interpretation = i + 1
        if "AUXILIARY TEMPORAL-DATA EXPERIMENTS" in text.upper():
            auxiliary_temporal = i + 1
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
    assert fine_tuning is not None, "TabPFN fine-tuning appendix not found"
    assert scratch_transformer is not None, "Scratch Transformer comparison appendix not found"
    assert interpretation is not None, "TabPFN interpretation appendix not found"
    assert auxiliary_temporal is not None, "Auxiliary temporal-data appendix not found"
    assert appendix < diagnostics < fine_tuning < scratch_transformer < interpretation < auxiliary_temporal, "Appendix order changed"
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
              "fine_tuning_start": fine_tuning, "scratch_transformer_start": scratch_transformer,
              "tabpfn_interpretation_start": interpretation,
              "auxiliary_temporal_start": auxiliary_temporal,
              "resolved_bibliography_entries": len(bibkeys),
              "page_bounds_and_reference_checks": "passed", "pages": details}
    (ROOT / "reports/final_report/pdf_verification.json").write_text(json.dumps(record,indent=2) + "\n",encoding="utf-8")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
