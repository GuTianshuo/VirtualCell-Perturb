"""Render docs/report.md to docs/report.pdf (Markdown -> HTML -> headless Chromium).

Requires:  pip install markdown playwright && python -m playwright install chromium
    python docs/build_report_pdf.py
"""
import os
import pathlib
import re

import markdown
from playwright.sync_api import sync_playwright

DOCS = pathlib.Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 10.5pt;
       line-height: 1.6; color: #1a1f26; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14.5pt; margin: 20pt 0 6pt; border-bottom: 1px solid #d9dfe6; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; } h1 + h3 { color: #4a5561; font-weight: 500; margin-top: 0; }
p, li { text-align: justify; } code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 9pt; background: #eef2f6; padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #d9dfe6; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f2f5f8; font-weight: 600; }
img { display: block; max-width: 100%; margin: 8pt auto 2pt; page-break-inside: avoid; }
img + em, p > em:only-child { display: block; text-align: center; color: #4a5561; font-size: 9pt; margin-bottom: 8pt; }
blockquote { color: #4a5561; border-left: 3px solid #d9dfe6; margin: 0; padding-left: 10pt; }
"""


def main():
    md = (DOCS / "report.md").read_text(encoding="utf-8")
    # protect TeX from the Markdown converter (underscores, backslashes), restore afterwards
    math = []

    def keep(m):
        math.append(m.group(0))
        return f"@@MATH{len(math) - 1}@@"
    md = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", keep, md, flags=re.S)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"@@MATH(\d+)@@", lambda m: math[int(m.group(1))].replace("<", "&lt;"), body)
    k = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
    head = (f'<link rel="stylesheet" href="{k}/katex.min.css"><script src="{k}/katex.min.js"></script>'
            f'<script src="{k}/contrib/auto-render.min.js"></script>')
    html = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><base href="{DOCS.as_uri()}/">{head}'
            f'<style>{CSS}</style></head><body>{body}</body></html>')
    tmp = DOCS / "_report_tmp.html"
    tmp.write_text(html, encoding="utf-8")
    try:
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception:
                # no bundled Chromium: use an installed Chrome/Edge (common on Windows)
                try:
                    b = p.chromium.launch(channel="chrome")
                except Exception:
                    b = p.chromium.launch(channel="msedge")
            pg = b.new_page()
            pg.goto(tmp.as_uri(), wait_until="load")
            pg.evaluate("""() => renderMathInElement(document.body, {delimiters: [
                {left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], throwOnError: false})""")
            left = pg.evaluate("document.body.innerText.split('$').length - 1")
            print("unrendered $ signs:", left)
            pg.pdf(path=str(DOCS / "report.pdf"), format="A4", print_background=True,
                   margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                   display_header_footer=True, header_template="<span></span>",
                   footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#8a929b">'
                                   '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
            b.close()
    finally:
        os.remove(tmp)
    print("wrote", DOCS / "report.pdf")


if __name__ == "__main__":
    main()
