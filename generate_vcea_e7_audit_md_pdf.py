from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import subprocess

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "outputs" / "tgrs_2019_2023" / "vcea_extra_experiments" / "E7_pan_europe_locked_audit"
CSV_DIR = BASE_DIR / "csv"
MANIFEST_DIR = BASE_DIR / "manifests"
FIG_DIR = BASE_DIR / "figures"

MD_PATH = BASE_DIR / "VCEA_E7_Pan_European_Locked_Audit_表格图片汇总.md"
HTML_PATH = BASE_DIR / "VCEA_E7_Pan_European_Locked_Audit_表格图片汇总.html"
PDF_PATH = BASE_DIR / "VCEA_E7_Pan_European_Locked_Audit_表格图片汇总_MD版.pdf"

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_edge() -> Path:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Microsoft Edge not found")


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.astype(object).itertuples(index=False, name=None):
        cells = []
        for value in row:
            text = "" if pd.isna(value) else str(value)
            text = text.replace("\n", "<br>")
            text = text.replace("|", "\\|")
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def dataframe_to_html(df: pd.DataFrame) -> str:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else str(x).replace("\n", "<br>"))
    return safe.to_html(index=False, escape=False, border=0, classes="data-table")


def build_markdown(table_files: list[Path], image_files: list[Path]) -> str:
    lines: list[str] = []
    lines.append("# VCEA E7 Pan-European Locked Audit 表格与图片汇总")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 说明")
    lines.append("- 本文档将 `csv/` 与 `manifests/` 中的全部表格，连同 `figures/` 中的全部图片，汇总到同一份 Markdown 文档。")
    lines.append("- 由于部分表格列较多，Markdown 里保留完整字段；PDF 导出时会使用横向页面和紧凑样式。")
    lines.append("")
    lines.append("## 目录")
    lines.append("")
    lines.append("### 表格")
    for idx, path in enumerate(table_files, start=1):
        lines.append(f"- T{idx:02d}: `{path.relative_to(BASE_DIR).as_posix()}`")
    lines.append("")
    lines.append("### 图片")
    for idx, path in enumerate(image_files, start=1):
        lines.append(f"- F{idx:02d}: `{path.relative_to(BASE_DIR).as_posix()}`")
    lines.append("")

    for idx, path in enumerate(table_files, start=1):
        df = pd.read_csv(path)
        lines.append(f"## T{idx:02d}. {path.name}")
        lines.append("")
        lines.append(f"来源：`{path.relative_to(BASE_DIR).as_posix()}`")
        lines.append("")
        lines.append(markdown_table(df))
        lines.append("")

    for idx, path in enumerate(image_files, start=1):
        rel = path.relative_to(BASE_DIR).as_posix()
        lines.append(f"## F{idx:02d}. {path.name}")
        lines.append("")
        lines.append(f"来源：`{rel}`")
        lines.append("")
        lines.append(f"![{path.name}]({rel})")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_html(table_files: list[Path], image_files: list[Path]) -> str:
    css = """
    @page {
      size: A3 landscape;
      margin: 10mm;
    }
    body {
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      color: #111;
      line-height: 1.35;
      font-size: 11px;
      margin: 0;
      padding: 0;
    }
    .page {
      page-break-after: always;
      padding: 4mm 2mm;
    }
    h1, h2, h3 {
      margin: 0 0 8px 0;
      color: #0b2e59;
    }
    .meta {
      color: #555;
      margin-bottom: 8px;
    }
    .toc ul {
      margin-top: 4px;
    }
    .data-table {
      border-collapse: collapse;
      width: 100%;
      table-layout: auto;
      font-size: 8px;
      word-break: break-word;
    }
    .data-table th,
    .data-table td {
      border: 1px solid #cfd8e3;
      padding: 3px 4px;
      vertical-align: top;
    }
    .data-table th {
      background: #d9e8f5;
      font-weight: 700;
      position: sticky;
      top: 0;
    }
    .data-table tr:nth-child(even) td {
      background: #f7f7f7;
    }
    .figure-wrap {
      page-break-inside: avoid;
    }
    .figure-wrap img {
      width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }
    .source {
      color: #555;
      font-size: 10px;
      margin: 2px 0 10px 0;
    }
    """
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{escape(MD_PATH.stem)}</title>")
    parts.append(f"<style>{css}</style></head><body>")

    parts.append("<section class='page'>")
    parts.append("<h1>VCEA E7 Pan-European Locked Audit 表格与图片汇总</h1>")
    parts.append(f"<div class='meta'>生成时间：{escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</div>")
    parts.append("<p>本 PDF 由 Markdown 版汇总文档衍生，表格来自 <code>csv/</code> 与 <code>manifests/</code>，图片来自 <code>figures/</code>。</p>")
    parts.append("<div class='toc'><h2>目录</h2>")
    parts.append("<h3>表格</h3><ul>")
    for idx, path in enumerate(table_files, start=1):
        parts.append(f"<li>T{idx:02d}. {escape(path.relative_to(BASE_DIR).as_posix())}</li>")
    parts.append("</ul><h3>图片</h3><ul>")
    for idx, path in enumerate(image_files, start=1):
        parts.append(f"<li>F{idx:02d}. {escape(path.relative_to(BASE_DIR).as_posix())}</li>")
    parts.append("</ul></div></section>")

    for idx, path in enumerate(table_files, start=1):
        df = pd.read_csv(path)
        parts.append("<section class='page'>")
        parts.append(f"<h2>T{idx:02d}. {escape(path.name)}</h2>")
        parts.append(f"<div class='source'>来源：{escape(path.relative_to(BASE_DIR).as_posix())}</div>")
        parts.append(dataframe_to_html(df))
        parts.append("</section>")

    for idx, path in enumerate(image_files, start=1):
        parts.append("<section class='page figure-wrap'>")
        parts.append(f"<h2>F{idx:02d}. {escape(path.name)}</h2>")
        parts.append(f"<div class='source'>来源：{escape(path.relative_to(BASE_DIR).as_posix())}</div>")
        parts.append(f"<img src='{path.resolve().as_uri()}' alt='{escape(path.name)}'>")
        parts.append("</section>")

    parts.append("</body></html>")
    return "".join(parts)


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    edge = find_edge()
    cmd = [
        str(edge),
        "--headless",
        "--disable-gpu",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Edge export failed: {result.stderr or result.stdout}")


def main() -> None:
    table_files = sorted(list(CSV_DIR.glob("*.csv")) + list(MANIFEST_DIR.glob("*.csv")))
    image_files = sorted(FIG_DIR.glob("*.png"))

    md_text = build_markdown(table_files, image_files)
    MD_PATH.write_text(md_text, encoding="utf-8")

    html_text = build_html(table_files, image_files)
    HTML_PATH.write_text(html_text, encoding="utf-8")

    export_pdf(HTML_PATH, PDF_PATH)
    print(f"Markdown written: {MD_PATH}")
    print(f"HTML written: {HTML_PATH}")
    print(f"PDF written: {PDF_PATH}")


if __name__ == "__main__":
    main()
