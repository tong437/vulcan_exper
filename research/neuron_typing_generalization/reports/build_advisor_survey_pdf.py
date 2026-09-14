#!/usr/bin/env python3
"""Render the advisor neuron-method survey Markdown as a styled PDF."""

from __future__ import annotations

import argparse
import io
import html
import re
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml
from matplotlib.figure import Figure
from weasyprint import HTML


DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def latex_to_svg(latex: str) -> str:
    """Render display math as inline SVG so fractions and limits remain legible."""
    latex = re.sub(r"\s+", " ", latex).strip()
    figure = Figure(figsize=(1, 1))
    figure.patch.set_alpha(0)
    figure.text(0, 0, f"${latex}$", fontsize=12, color="#111827")
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0.03, transparent=True)
    svg = buffer.getvalue()
    svg = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!DOCTYPE.*?>", "", svg, flags=re.DOTALL)
    return svg.strip()


def protect_math(source: str) -> tuple[str, dict[str, str]]:
    """Replace LaTeX fragments with placeholders before Markdown conversion."""
    replacements: dict[str, str] = {}

    def display(match: re.Match[str]) -> str:
        key = f"MATHBLOCK{len(replacements):05d}TOKEN"
        latex = match.group(1).strip()
        replacements[key] = f'<div class="math-block">{latex_to_svg(latex)}</div>'
        return f"\n\n{key}\n\n"

    def inline(match: re.Match[str]) -> str:
        key = f"MATHINLINE{len(replacements):05d}TOKEN"
        latex = match.group(1).strip()
        replacements[key] = f'<span class="math-inline">{latex_to_mathml(latex)}</span>'
        return key

    protected = DISPLAY_MATH_RE.sub(display, source)
    protected = INLINE_MATH_RE.sub(inline, protected)
    return protected, replacements


def restore_math(rendered: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_html(markdown_path: Path, font_path: Path) -> str:
    source = markdown_path.read_text(encoding="utf-8")
    protected, replacements = protect_math(source)
    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    body = restore_math(body, replacements)

    font_uri = font_path.resolve().as_uri()
    title = "多模态大模型神经元重要性与分类方法调研"
    css = f"""
    @font-face {{
      font-family: 'Noto CJK SC';
      src: url('{font_uri}');
      font-weight: 100 900;
    }}
    @page {{
      size: A4;
      margin: 18mm 17mm 20mm 17mm;
      @bottom-left {{
        content: '神经元重要性与分类方法调研';
        color: #64748b;
        font-size: 8pt;
      }}
      @bottom-right {{
        content: counter(page) ' / ' counter(pages);
        color: #64748b;
        font-size: 8pt;
      }}
    }}
    * {{ box-sizing: border-box; }}
    html {{ font-family: 'Noto CJK SC', sans-serif; color: #172033; }}
    body {{ font-size: 10pt; line-height: 1.62; margin: 0; }}
    h1 {{
      color: #0f2f57;
      font-size: 24pt;
      line-height: 1.25;
      margin: 20mm 0 10mm;
      padding-bottom: 6mm;
      border-bottom: 2.5pt solid #1d4ed8;
      text-align: left;
    }}
    h1 + p {{
      background: #eef5ff;
      border-left: 4pt solid #2563eb;
      padding: 4mm 5mm;
      color: #334155;
      margin-bottom: 9mm;
    }}
    h2 {{
      color: #163f73;
      font-size: 16pt;
      line-height: 1.35;
      margin: 8mm 0 4mm;
      padding-bottom: 1.5mm;
      border-bottom: 0.7pt solid #bfd3ef;
      break-after: avoid;
    }}
    h3 {{
      color: #1f4f82;
      font-size: 12.2pt;
      margin: 5mm 0 2mm;
      break-after: avoid;
    }}
    h4 {{ color: #334155; font-size: 10.5pt; break-after: avoid; }}
    p {{ margin: 0 0 3.2mm; orphans: 3; widows: 3; }}
    ul, ol {{ margin: 1.5mm 0 3.5mm 5.5mm; padding-left: 4mm; }}
    li {{ margin: 0 0 1.2mm; }}
    strong {{ color: #0f2744; }}
    code {{
      font-family: 'DejaVu Sans Mono', monospace;
      font-size: 8.8pt;
      background: #f1f5f9;
      color: #9f1239;
      padding: 0.2mm 1mm;
      border-radius: 1mm;
    }}
    a {{ color: #155dcc; text-decoration: underline; text-decoration-color: #93b4e6; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 3mm 0 5mm;
      font-size: 8.7pt;
      break-inside: avoid;
    }}
    thead {{ display: table-header-group; }}
    th {{
      background: #163f73;
      color: white;
      font-weight: 600;
      text-align: left;
      padding: 2.2mm 2.5mm;
      border: 0.5pt solid #163f73;
    }}
    td {{
      padding: 2mm 2.5mm;
      border: 0.5pt solid #cad5e2;
      vertical-align: top;
    }}
    tbody tr:nth-child(even) {{ background: #f7faff; }}
    .math-block {{
      text-align: center;
      margin: 3.5mm auto;
      padding: 2mm 1mm;
      color: #111827;
      break-inside: avoid;
      font-size: 10.5pt;
    }}
    .math-block svg {{ max-width: 100%; height: auto; max-height: 18mm; }}
    .math-inline {{ white-space: nowrap; }}
    math {{ font-family: 'DejaVu Serif', serif; }}
    blockquote {{
      margin: 4mm 0;
      padding: 3mm 4mm;
      border-left: 3pt solid #3b82f6;
      background: #f5f9ff;
      color: #334155;
    }}
    .decision-box {{
      margin: 3mm auto 4mm;
      padding: 3mm 5mm;
      width: 72%;
      text-align: center;
      color: #123f73;
      background: #eef5ff;
      border: 1pt solid #82aee7;
      border-radius: 2mm;
      break-inside: avoid;
    }}
    h2:nth-of-type(3), h2:nth-of-type(7), h2:nth-of-type(12) {{ break-before: page; }}
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="author" content="Neuron Typing Generalization Research">
  <meta name="subject" content="神经元重要性计算与分类方案">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>{body}</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()

    html_text = build_html(args.source.resolve(), args.font.resolve())
    if args.html_output:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(html_text, encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_text, base_url=str(args.source.parent.resolve())).write_pdf(
        args.output,
        metadata={
            "Title": "多模态大模型神经元重要性与分类方法调研",
            "Author": "Neuron Typing Generalization Research",
            "Subject": "导师讨论稿",
        },
    )


if __name__ == "__main__":
    main()
