from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "outputs" / "tgrs_2019_2023" / "vcea_extra_experiments" / "E7_pan_europe_locked_audit"
CSV_DIR = BASE_DIR / "csv"
MANIFEST_DIR = BASE_DIR / "manifests"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_PDF = BASE_DIR / "VCEA_E7_Pan_European_Locked_Audit_表格图片汇总.pdf"


def configure_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def page_size(landscape: bool) -> tuple[float, float]:
    return (11.69, 8.27) if landscape else (8.27, 11.69)


def wrap_text(value: object, width: int) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        lines.append(textwrap.fill(raw, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


def add_cover(pdf: PdfPages, font_name: str, table_files: list[Path], image_files: list[Path]) -> None:
    fig = plt.figure(figsize=page_size(False))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    lines = [
        "VCEA E7 Pan-European Locked Audit",
        "表格与图片完整汇总",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"字体: {font_name}",
        f"表格数量: {len(table_files)}",
        f"图片数量: {len(image_files)}",
        "",
        "内容说明:",
        "1. csv/ 与 manifests/ 中的全部 CSV 表格完整分页收录",
        "2. figures/ 中的全部 PNG 图片整页收录",
        "3. 长表自动分页，避免内容截断",
        "",
        "实验目录:",
        str(BASE_DIR),
    ]
    y = 0.92
    for i, line in enumerate(lines):
        fontsize = 22 if i == 0 else 17 if i == 1 else 11
        ax.text(0.08, y, line, fontsize=fontsize, va="top")
        y -= 0.055 if i < 2 else 0.038
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_index(pdf: PdfPages, table_files: list[Path], image_files: list[Path]) -> None:
    fig = plt.figure(figsize=page_size(False))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.95, "目录", fontsize=20, va="top")
    y = 0.90
    ax.text(0.06, y, "表格", fontsize=14, va="top")
    y -= 0.035
    for idx, path in enumerate(table_files, start=1):
        ax.text(0.08, y, f"T{idx:02d}. {path.relative_to(BASE_DIR).as_posix()}", fontsize=10, va="top")
        y -= 0.028
    y -= 0.01
    ax.text(0.06, y, "图片", fontsize=14, va="top")
    y -= 0.035
    for idx, path in enumerate(image_files, start=1):
        ax.text(0.08, y, f"F{idx:02d}. {path.relative_to(BASE_DIR).as_posix()}", fontsize=10, va="top")
        y -= 0.028
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=page_size(False))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.text(0.06, 0.95, "目录（续）", fontsize=20, va="top")
            y = 0.90
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_pages(pdf: PdfPages, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    ncols = len(df.columns)
    landscape = ncols > 6
    figsize = page_size(landscape)
    wrap_width = 14 if ncols >= 8 else 18 if ncols >= 6 else 24
    rows_per_page = 12 if ncols >= 8 else 16 if ncols >= 6 else 20

    wrapped_columns = [wrap_text(col, wrap_width) for col in df.columns]
    wrapped_rows = [[wrap_text(value, wrap_width) for value in row] for row in df.astype(object).itertuples(index=False, name=None)]
    total_pages = max(1, math.ceil(max(len(wrapped_rows), 1) / rows_per_page))
    if not wrapped_rows:
        wrapped_rows = [[""] * ncols]

    for page_idx in range(total_pages):
        start = page_idx * rows_per_page
        end = start + rows_per_page
        chunk = wrapped_rows[start:end]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
        ax.axis("off")
        fig.text(0.03, 0.96, csv_path.relative_to(BASE_DIR).as_posix(), fontsize=15, va="top")
        fig.text(0.03, 0.92, f"表格页 {page_idx + 1}/{total_pages} | 行 {start + 1}-{min(end, len(wrapped_rows))} / {len(wrapped_rows)}", fontsize=10, va="top")

        table = ax.table(cellText=chunk, colLabels=wrapped_columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.4 if ncols >= 8 else 7.2 if ncols >= 6 else 8.4)
        table.scale(1, 1.15)

        base_h = 0.035 if landscape else 0.04
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            cell.get_text().set_wrap(True)
            text = cell.get_text().get_text()
            line_count = max(1, text.count("\n") + 1)
            if row == 0:
                cell.set_facecolor("#d9e8f5")
                cell.set_text_props(weight="bold")
                cell.set_height(base_h * max(1.3, line_count * 0.9))
            else:
                cell.set_height(base_h * max(1.0, line_count * 0.82))
                if row % 2 == 0:
                    cell.set_facecolor("#f7f7f7")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, index: int, total: int) -> None:
    with Image.open(image_path) as img:
        width, height = img.size
        landscape = width >= height
        fig = plt.figure(figsize=page_size(landscape))
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
        ax.axis("off")
        fig.text(0.03, 0.96, image_path.relative_to(BASE_DIR).as_posix(), fontsize=15, va="top")
        fig.text(0.03, 0.92, f"图片 {index}/{total}", fontsize=10, va="top")
        ax.imshow(img)
        ax.set_aspect("auto")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    font_name = configure_font()
    table_files = sorted(list(CSV_DIR.glob("*.csv")) + list(MANIFEST_DIR.glob("*.csv")))
    image_files = sorted(FIG_DIR.glob("*.png"))
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        add_cover(pdf, font_name, table_files, image_files)
        add_index(pdf, table_files, image_files)
        for csv_path in table_files:
            add_table_pages(pdf, csv_path)
        for idx, image_path in enumerate(image_files, start=1):
            add_image_page(pdf, image_path, idx, len(image_files))

    print(f"PDF generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
