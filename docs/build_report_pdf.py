"""Inline the report's figures as data URIs (the PDF renderer loads HTML from memory, not from disk)
and write docs/_report_inlined.md, a self-contained Markdown file. The PDF in this folder was rendered from it
with Chromium's printToPDF (A4, 12 pt Times New Roman, 1.5 line height); any Markdown-to-PDF tool
(e.g. pandoc) gives an equivalent document.
"""
import base64, re
from pathlib import Path
src = Path(__file__).with_name("technical_report.md")
s = src.read_text(encoding="utf-8")
def rep(m):
    p = (src.parent / m.group(2)).resolve()
    return f"![{m.group(1)}](data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()})"
s = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", rep, s)
src.with_name("_report_inlined.md").write_text(s, encoding="utf-8")
print("ok")
