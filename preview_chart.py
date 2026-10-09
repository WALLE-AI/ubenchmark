#!/usr/bin/env python3
"""把报告里的折线图单独渲成 PNG，用来肉眼检查几何、标签碰撞、溢出。

校验器只管颜色，版面得自己看。用法:
  PY preview_chart.py [结果JSON] [profile] [out.png]
"""

import json
import sys

import fitz

import ocr_report_html as R


def svg_to_png(svg_body, out, scale=2, dark=False):
    bg = R.PALETTE_DARK["surface"] if dark else R.PALETTE_LIGHT["surface"]
    inner = svg_body[svg_body.index(">") + 1:]
    doc_svg = (f"<svg xmlns='http://www.w3.org/2000/svg' "
               f"width='{R.VB_W * scale}' height='{R.VB_H * scale}' "
               f"viewBox='0 0 {R.VB_W} {R.VB_H}'>"
               f"<rect width='100%' height='100%' fill='{bg}'/>{inner}")
    tmp = out + ".svg"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc_svg)
    doc = fitz.open(tmp)
    doc.load_page(0).get_pixmap(dpi=200).save(out)
    doc.close()
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "ocr_e2e_report.json"
    profile = sys.argv[2] if len(sys.argv) > 2 else None
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/preview"

    report = json.load(open(src, encoding="utf-8"))
    profiles = [p["profile"] for p in report["targets"][0]["profiles"]]
    profile = profile or profiles[0]
    charts = R.build_charts(report, profiles)
    thresh = report["p99_threshold_ms"] / 1000.0

    for mode, pal, dark in (("light", R.PALETTE_LIGHT, False),
                            ("dark", R.PALETTE_DARK, True)):
        fps = R.line_chart_svg(charts["fps"][profile], pal,
                               y_tick_fmt=lambda t: f"{t:.1f}", label_peak=True)
        p99 = R.line_chart_svg(charts["p99"][profile], pal,
                               y_tick_fmt=lambda t: f"{t:.0f}", threshold=thresh)
        svg_to_png(fps, f"{out}_fps_{mode}.png", dark=dark)
        svg_to_png(p99, f"{out}_p99_{mode}.png", dark=dark)
        print(f"{mode}: {out}_fps_{mode}.png  {out}_p99_{mode}.png")


if __name__ == "__main__":
    main()
