#!/usr/bin/env python3
"""把 bench_ocr_e2e.py 的结果 JSON 渲染成自包含的图文 HTML 报告。

图表是 Python 直接生成的静态 SVG (禁用 JS 也能看)，JS 只负责叠加十字准星和悬停读数。
配色取自已验证的分类色板 (槽1 blue = mineru, 槽2 orange = teleocr)，
明暗两套均通过 CVD 分离度、对比度等六项校验。
"""

import json
import math

# 明暗两套色板均已过校验器 (validate_palette.js)，勿随手改十六进制值。
PALETTE_LIGHT = {
    "surface": "#fcfcfb", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "muted": "#898781", "secondary": "#52514e",
    "series": ["#2a78d6", "#eb6834"], "critical": "#d03b3b",
}
PALETTE_DARK = {
    "surface": "#1a1a19", "grid": "#2c2c2a", "axis": "#383835",
    "muted": "#898781", "secondary": "#c3c2b7",
    "series": ["#3987e5", "#d95926"], "critical": "#d03b3b",
}
# 报告里走 CSS 变量，明暗切换只改一处；栅格化预览时换成上面的具体色值。
PALETTE_VAR = {
    "surface": "var(--surface-1)", "grid": "var(--grid)", "axis": "var(--axis)",
    "muted": "var(--text-muted)", "secondary": "var(--text-secondary)",
    "series": ["var(--series-1)", "var(--series-2)"], "critical": "var(--critical)",
}

PROFILE_SUB = {
    "tiny_short": "&lt;0.2MB · 1–5 页",
    "small_medium": "0.2–2MB · 6–30 页",
    "medium_long": "2–10MB · 31–100 页",
    "large_long": "10–50MB · 31–100 页",
    "xlarge_verylong": "&gt;50MB · 100+ 页",
}

VB_W, VB_H = 340, 218
MARGIN = {"t": 14, "r": 18, "b": 30, "l": 48}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def nice_ticks(vmax, count=4):
    if vmax <= 0:
        return [0.0, 1.0]
    raw = vmax / count
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm <= 1 else 2 if norm <= 2 else 5 if norm <= 5 else 10) * mag
    ticks, v = [], 0.0
    while v <= vmax + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    if ticks[-1] < vmax:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def line_chart_svg(spec, pal, y_tick_fmt, threshold=None, label_peak=False):
    """一个小面板的折线图。并发是 1,2,4,…,64 的几何序列，用 log2 轴等距展开，
    否则低并发全挤在左端看不出拐点。"""
    xs = spec["xTicks"]
    pw = VB_W - MARGIN["l"] - MARGIN["r"]
    ph = VB_H - MARGIN["t"] - MARGIN["b"]
    lx0, lx1 = math.log2(xs[0]), math.log2(xs[-1])

    def X(v):
        return MARGIN["l"] + (math.log2(v) - lx0) / (lx1 - lx0) * pw

    vmax = max([p["y"] for s in spec["series"] for p in s["points"] if p["y"] is not None]
               or [1.0])
    if threshold is not None:
        vmax = max(vmax, threshold)
    ticks = nice_ticks(vmax)
    top = ticks[-1] or 1.0

    def Y(v):
        return MARGIN["t"] + ph - (v / top) * ph

    o = [f"<svg viewBox='0 0 {VB_W} {VB_H}' width='100%' "
         f"preserveAspectRatio='xMidYMid meet' role='img' "
         f"aria-label='{esc(spec['aria'])}' class='chart'>"]

    # 网格线: 实线发丝线，比表面深一档，保持后退
    for t in ticks:
        color = pal["axis"] if t == 0 else pal["grid"]
        o.append(f"<line x1='{MARGIN['l']}' x2='{MARGIN['l'] + pw}' "
                 f"y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='{color}' stroke-width='1'/>")
        o.append(f"<text x='{MARGIN['l'] - 8}' y='{Y(t) + 3.6:.1f}' text-anchor='end' "
                 f"fill='{pal['muted']}' font-size='10.5' "
                 f"style='font-variant-numeric:tabular-nums'>{y_tick_fmt(t)}</text>")

    for v in xs:
        o.append(f"<text x='{X(v):.1f}' y='{VB_H - 10}' text-anchor='middle' "
                 f"fill='{pal['muted']}' font-size='10.5' "
                 f"style='font-variant-numeric:tabular-nums'>{v}</text>")

    # 阈值参考线: 这里的虚线是语义上的阈值标注，不是网格线
    if threshold is not None and threshold <= top:
        o.append(f"<line x1='{MARGIN['l']}' x2='{MARGIN['l'] + pw}' "
                 f"y1='{Y(threshold):.1f}' y2='{Y(threshold):.1f}' "
                 f"stroke='{pal['critical']}' stroke-width='1' "
                 f"stroke-dasharray='4 3' opacity='0.75'/>")

    marked = []
    for i, s in enumerate(spec["series"]):
        color = pal["series"][i % len(pal["series"])]
        pts = [p for p in s["points"] if p["y"] is not None]
        if not pts:
            continue
        d = " ".join(f"{'L' if k else 'M'}{X(p['x']):.1f},{Y(p['y']):.1f}"
                     for k, p in enumerate(pts))
        o.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' "
                 f"stroke-linejoin='round' stroke-linecap='round'/>")
        for p in pts:
            rec = p["x"] == s.get("recommended")
            # 2px 表面色环，让点与线/彼此交叠时仍然分得开
            o.append(f"<circle cx='{X(p['x']):.1f}' cy='{Y(p['y']):.1f}' "
                     f"r='{5.5 if rec else 4}' fill='{color}' "
                     f"stroke='{pal['surface']}' stroke-width='2'/>")
            if rec:
                o.append(f"<circle cx='{X(p['x']):.1f}' cy='{Y(p['y']):.1f}' r='8.5' "
                         f"fill='none' stroke='{color}' stroke-width='1.5' opacity='0.55'/>")
        rp = next((p for p in pts if p["x"] == s.get("recommended")), None)
        if rp:
            marked.append((X(rp["x"]), Y(rp["y"]), rp["y"]))

    # 只给"建议并发"这一个点标数值 —— 它是可执行的那个结论，且已有圆环指示。
    # 不标峰值: 峰值常和建议点不在同一处，两个标注会互相competing，还容易压到另一条线上。
    if label_peak and marked:
        placed = []
        for px, py, val in sorted(marked, key=lambda z: z[1]):
            ty = py - 13
            while any(abs(ty - q) < 12 for q in placed):
                ty -= 12
            placed.append(ty)
            anchor, tx = (("end", px + 8) if px > MARGIN["l"] + pw * 0.72
                          else ("middle", px))
            ty = max(ty, 10)
            common = (f"x='{tx:.1f}' y='{ty:.1f}' text-anchor='{anchor}' "
                      f"font-size='10.5' font-weight='600' "
                      f"style='font-variant-numeric:tabular-nums'")
            # 文字压在线上时的底衬: 先画一份表面色描边的副本，再把实心字画在上面。
            # 不用 paint-order —— 不是所有 SVG 渲染器都支持，不支持时描边会盖掉字形。
            o.append(f"<text {common} fill='none' stroke='{pal['surface']}' "
                     f"stroke-width='3' stroke-linejoin='round'>{val:.2f}</text>")
            o.append(f"<text {common} fill='{pal['secondary']}'>{val:.2f}</text>")

    # 悬停热区。用 fill-opacity=0 而不是 fill='transparent' —— 后者不是所有 SVG
    # 渲染器都认，回退成黑色会把整个图糊掉 (MuPDF 就是这样)。
    o.append(f"<rect class='hit' x='{MARGIN['l']}' y='{MARGIN['t']}' "
             f"width='{pw}' height='{ph}' fill='#ffffff' fill-opacity='0' tabindex='0'/>")
    o.append(f"<line class='cross' y1='{MARGIN['t']}' y2='{MARGIN['t'] + ph}' "
             f"stroke='{pal['axis']}' stroke-width='1' opacity='0'/>")
    o.append("</svg>")
    return "".join(o)


CSS = """
.viz-root{
  color-scheme:light;
  --surface-1:#fcfcfb; --page:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --tile:#ffffff;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --page:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --series-1:#3987e5; --series-2:#d95926; --tile:#1f1f1e;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --tile:#1f1f1e;
}

*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font:14px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 64px}
h1{font-size:26px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;font-weight:600;margin:46px 0 6px}
h3{font-size:15px;font-weight:600;margin:28px 0 8px;color:var(--text-secondary)}
p{margin:0 0 4px}
.lede{color:var(--text-secondary)}
.muted{color:var(--text-muted);font-size:13px}
section>.muted{margin-bottom:14px}

.callout{background:var(--surface-1);border:1px solid var(--border);
  border-left:3px solid var(--warning);border-radius:4px;padding:14px 18px;margin:22px 0 8px}
.callout ul{margin:8px 0 0;padding-left:20px;color:var(--text-secondary)}
.callout li{margin:5px 0}
code{background:var(--page);border:1px solid var(--border);border-radius:3px;
  padding:1px 5px;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(198px,1fr));gap:14px;margin:18px 0 8px}
.tile{background:var(--tile);border:1px solid var(--border);border-radius:6px;padding:16px 18px}
.tile .k{color:var(--text-secondary);font-size:13px;margin-bottom:6px}
.tile .v{font-size:27px;font-weight:600;line-height:1.15;letter-spacing:-.015em}
.tile .u{font-size:14px;font-weight:400;color:var(--text-secondary);margin-left:3px}
.tile .s{color:var(--text-muted);font-size:12px;margin-top:5px}
.key{display:inline-block;width:11px;height:2px;border-radius:1px;
  vertical-align:middle;margin-right:6px;position:relative;top:-2px}

.card{background:var(--surface-1);border:1px solid var(--border);border-radius:6px;
  padding:18px 20px 14px;margin:14px 0}
.legend{display:flex;gap:20px;align-items:center;margin:0 0 10px;flex-wrap:wrap}
.legend .item{display:flex;align-items:center;gap:7px;color:var(--text-secondary);font-size:13px}
.legend .line{width:16px;height:2px;border-radius:1px}
.legend .ring{width:9px;height:9px;border-radius:50%;border:1.5px solid var(--text-muted)}
.facets{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px 14px}
.facet{position:relative;min-width:0}
.facet .ft{font-size:13px;font-weight:600;margin:0 0 1px;padding-left:10px}
.facet .ft small{display:block;font-weight:400;color:var(--text-muted);font-size:11.5px}
.chart{display:block;overflow:visible}
.hit{cursor:crosshair}
.hit:focus{outline:none}

.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s;
  background:var(--surface-1);border:1px solid var(--border);border-radius:5px;
  padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.16);z-index:20;min-width:126px}
.tip .th{color:var(--text-muted);font-size:11px;margin-bottom:5px}
.tip .r{display:flex;align-items:center;gap:7px;margin:3px 0}
.tip .r .line{width:11px;height:2px;border-radius:1px;flex:none}
.tip .r .v{font-weight:600;font-variant-numeric:tabular-nums}
.tip .r .n{color:var(--text-secondary)}

table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px;
  background:var(--surface-1);border:1px solid var(--border);border-radius:6px}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--border);
  font-variant-numeric:tabular-nums}
th{color:var(--text-secondary);font-weight:600;background:var(--page)}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
tbody tr:last-child td{border-bottom:none}
tr.rec td{background:color-mix(in srgb,var(--good) 10%,transparent);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;
  vertical-align:middle;position:relative;top:-1px}
.bad{color:var(--critical)}
details{margin:6px 0 16px}
summary{cursor:pointer;color:var(--text-secondary);font-size:13px;padding:5px 0}
footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--border);
  color:var(--text-muted);font-size:12.5px}
"""

# JS 只负责悬停读数; 图表本身是静态 SVG，禁用 JS 也完整可读
JS = r"""
const D = JSON.parse(document.getElementById('bench-data').textContent);
const VB_W = 340, ML = 48, MR = 18;

document.querySelectorAll('.facet[data-chart]').forEach(facet => {
  const svg = facet.querySelector('svg');
  const hit = svg.querySelector('.hit');
  const cross = svg.querySelector('.cross');
  const spec = D.charts[facet.dataset.chart][facet.dataset.profile];
  const isFps = facet.dataset.chart === 'fps';
  const fmt = isFps ? (v => v.toFixed(2) + ' 页/s') : (v => v.toFixed(1) + ' s');
  const xs = spec.xTicks;
  const lx0 = Math.log2(xs[0]), lx1 = Math.log2(xs[xs.length - 1]);
  const pw = VB_W - ML - MR;
  const X = v => ML + (Math.log2(v) - lx0) / (lx1 - lx0) * pw;

  const tip = document.createElement('div');
  tip.className = 'tip';
  facet.appendChild(tip);

  const show = clientX => {
    const box = svg.getBoundingClientRect();
    const vx = (clientX - box.left) / box.width * VB_W;
    let best = xs[0], bd = Infinity;
    for (const v of xs) { const d = Math.abs(X(v) - vx); if (d < bd) { bd = d; best = v; } }
    cross.setAttribute('x1', X(best)); cross.setAttribute('x2', X(best));
    cross.setAttribute('opacity', 0.55);

    tip.textContent = '';
    const th = document.createElement('div');
    th.className = 'th'; th.textContent = '并发 ' + best;
    tip.appendChild(th);
    spec.series.forEach((s, i) => {
      const p = s.points.find(q => q.x === best);
      const row = document.createElement('div'); row.className = 'r';
      const k = document.createElement('span'); k.className = 'line';
      k.style.background = 'var(--series-' + (i + 1) + ')'; row.appendChild(k);
      const v = document.createElement('span'); v.className = 'v';
      v.textContent = (p && p.y != null) ? fmt(p.y) : '—'; row.appendChild(v);
      /* 系列名是外部数据，用 textContent 而不是拼 innerHTML */
      const n = document.createElement('span'); n.className = 'n';
      n.textContent = s.name; row.appendChild(n);
      tip.appendChild(row);
    });
    tip.style.opacity = 1;
    const px = X(best) / VB_W * box.width;
    tip.style.left = Math.min(Math.max(px - 63, 0), box.width - 132) + 'px';
    tip.style.top = (svg.offsetTop + 8) + 'px';
  };
  const hide = () => { tip.style.opacity = 0; cross.setAttribute('opacity', 0); };

  hit.addEventListener('pointermove', e => show(e.clientX));
  hit.addEventListener('pointerleave', hide);
  hit.addEventListener('focus', () => {
    const box = svg.getBoundingClientRect();
    show(box.left + box.width / 2);
  });
  hit.addEventListener('blur', hide);
});
"""


def fmt_ms(v):
    return f"{v:.0f}" if v is not None else "—"


def fmt_hit(pc):
    return f"{pc['hit_rate'] * 100:.0f}%" if pc else "—"


def build_charts(report, profiles):
    charts = {"fps": {}, "p99": {}}
    for profile in profiles:
        for metric, key, scale in (("fps", "fps", 1.0), ("p99", "e2e_p99_ms", 0.001)):
            series = []
            for t in report["targets"]:
                pr = next((x for x in t["profiles"] if x["profile"] == profile), None)
                if not pr:
                    continue
                series.append({
                    "name": t["name"],
                    "recommended": pr["recommended_concurrency"],
                    "points": [
                        {"x": l["concurrency"],
                         "y": (l[key] * scale) if l[key] is not None else None}
                        for l in sorted(pr["levels"], key=lambda z: z["concurrency"])
                    ],
                })
            xs = sorted({p["x"] for s in series for p in s["points"]})
            charts[metric][profile] = {
                "series": series, "xTicks": xs,
                "aria": (f"{profile} 分档下两个模型的"
                         f"{'吞吐(页/秒)' if metric == 'fps' else 'p99 延迟(秒)'}"
                         f"随并发数变化的折线图"),
            }
    return charts


def peak_of(target):
    best = None
    for pr in target["profiles"]:
        for l in pr["levels"]:
            if best is None or l["fps"] > best[0]:
                best = (l["fps"], pr["profile"], l["concurrency"])
    return best


def rec_at(pr):
    return next((l for l in pr["levels"]
                 if l["concurrency"] == pr["recommended_concurrency"]), None)


def write_report(report, path, palette=None):
    pal = palette or PALETTE_VAR
    ds = report["dataset"]
    targets = report["targets"]
    profiles = [p["profile"] for p in targets[0]["profiles"]] if targets else []
    n_pages = sum(len(e["page_indices"]) for e in ds["entries"])
    charts = build_charts(report, profiles)
    thresh_s = report["p99_threshold_ms"] / 1000.0

    o = []
    a = o.append
    a("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>文档解析 VLM 端到端压测报告</title>")
    a(f"<style>{CSS}</style></head><body><div class='viz-root'><div class='wrap'>")

    a("<h1>文档解析 VLM 端到端页级压测报告</h1>")
    a("<p class='lede'>MinerU2.5-Pro-1.2B vs TeleOCR · 真实 PDF 语料分层抽样 · "
      "两阶段原生协议（1 次 layout 请求 + N 次版面块请求 = 1 页）</p>")
    a(f"<p class='muted'>生成于 {esc(report['generated_at'])} · "
      f"语料 {ds['catalog_total']} 份中抽样 {len(ds['entries'])} 篇 / {n_pages} 页 · "
      f"并发梯度 {esc(report['concurrency_levels'])}</p>")

    # ---- 关键读数 ----
    a("<section><h2>关键读数</h2><div class='tiles'>")
    for i, t in enumerate(targets):
        peak = peak_of(t)
        recs = [pr["recommended_concurrency"] for pr in t["profiles"]
                if pr["recommended_concurrency"]]
        rng = f"{min(recs)}–{max(recs)}" if recs else "—"
        a("<div class='tile'>"
          f"<div class='k'><span class='key' style='background:var(--series-{i + 1})'></span>"
          f"{esc(t['name'])} 峰值吞吐</div>"
          f"<div class='v'>{peak[0]:.2f}<span class='u'>页/s</span></div>"
          f"<div class='s'>{esc(peak[1])} @ 并发 {peak[2]} · 建议并发 {rng}</div></div>")
    avg_b = sum(t["avg_blocks_per_page"] for t in targets) / max(len(targets), 1)
    total_req = sum(l["requests_total"] for t in targets
                    for pr in t["profiles"] for l in pr["levels"])
    a("<div class='tile'><div class='k'>每页请求放大</div>"
      f"<div class='v'>{avg_b + 1:.1f}<span class='u'>请求/页</span></div>"
      f"<div class='s'>1 次 layout + 平均 {avg_b:.1f} 个版面块</div></div>")
    a("<div class='tile'><div class='k'>压测总请求</div>"
      f"<div class='v'>{total_req / 1000:.1f}<span class='u'>K</span></div>"
      f"<div class='s'>覆盖 {len(profiles)} 个 大小×页数 分档</div></div>")
    a("</div></section>")

    # ---- 读数前提 ----
    a("<div class='callout'><b>读数前提</b><ul>")
    a("<li>两个服务<b>显存配额不等</b>：mineru 在 GPU0 <code>gpu-memory-utilization=0.3</code>"
      "（约 12GB）、teleocr 在 GPU1 <code>gpu-memory-utilization=0.25</code>（约 10GB）。"
      "属非等配额对比，不能据此对两个模型的绝对性能下定论。</li>")
    if report.get("cache_busting", True):
        a("<li>每个请求实例的图片都打了唯一像素水印，必然 miss vLLM 前缀缓存，对齐"
          "「生产里每页都是新的」。表中前缀缓存命中率应接近 0%；若显著偏高说明去重失效。"
          "（不去重时实测命中率可达 96%，吞吐虚高 15–40%。）</li>")
    else:
        a("<li><b>已关闭图片水印去重</b>：重放同一批页面会大量命中前缀缓存，"
          "此报告是缓存全命中时的上限吞吐，<b>不能用于容量规划</b>。</li>")
    a("<li>客户端连接池上限 256，并对连接级错误重试 1 次。不加限制时 C=64 会瞬间要开近千条 "
      "TCP 连接，触发 <code>Connection reset by peer</code> —— 那测的是连接层而非推理算力。</li>")
    a("<li>显存列是目标 GPU 的<b>整卡</b>占用（含卡上其他进程），不是本服务的独占用量。</li>")
    a("<li>官方客户端的 <code>no_repeat_ngram_size=100</code> 在 vLLM OpenAI 接口不支持，"
      "已省略；其余采样参数与官方一致。</li>")
    a("</ul></div>")

    # ---- 建议并发 ----
    a("<section><h2>建议最大支撑并发</h2>")
    tol = report.get("fps_tolerance", report.get("fps_gain_threshold", 0.05))
    a(f"<p class='muted'>判定：先筛出达标档（单页端到端 p99 ≤ {thresh_s:.0f}s "
      f"且错误率 ≤ {report['error_rate_threshold']:.0%}），再取其中"
      f"<b>最小的、吞吐已达峰值 {1 - tol:.0%} 的并发</b> —— 越过这个拐点，"
      f"加并发只换来不到 {tol:.0%} 的吞吐，却要线性地付延迟。</p>")
    a("<table><thead><tr><th>分档</th>")
    for i, t in enumerate(targets):
        a(f"<th><span class='dot' style='background:var(--series-{i + 1})'></span>"
          f"{esc(t['name'])} 并发</th><th>吞吐</th><th style='text-align:left'>判定依据</th>")
    a("</tr></thead><tbody>")
    for profile in profiles:
        a(f"<tr><td>{esc(profile)}</td>")
        for t in targets:
            pr = next((x for x in t["profiles"] if x["profile"] == profile), None)
            lvl = rec_at(pr) if pr else None
            if not lvl:
                a("<td>—</td><td>—</td><td style='text-align:left'>—</td>")
            else:
                a(f"<td>{pr['recommended_concurrency']}</td><td>{lvl['fps']:.2f} 页/s</td>"
                  f"<td class='muted' style='text-align:left;font-size:12px'>"
                  f"{esc(pr['recommendation_note'])}</td>")
        a("</tr>")
    a("</tbody></table></section>")

    # ---- 图表 ----
    legend = ["<div class='legend'>"]
    for i, t in enumerate(targets):
        legend.append(f"<span class='item'><span class='line' "
                      f"style='background:var(--series-{i + 1})'></span>{esc(t['name'])}</span>")
    legend.append("<span class='item'><span class='ring'></span>建议并发所在点</span>")
    legend.append("</div>")
    legend = "".join(legend)

    def facets(metric, **kw):
        rows = ["<div class='facets'>"]
        for profile in profiles:
            svg = line_chart_svg(charts[metric][profile], pal, **kw)
            rows.append(
                f"<div class='facet' data-chart='{metric}' data-profile='{esc(profile)}'>"
                f"<p class='ft'>{esc(profile)}"
                f"<small>{PROFILE_SUB.get(profile, '')}</small></p>{svg}</div>")
        rows.append("</div>")
        return "".join(rows)

    a("<section><h2>吞吐随并发的变化</h2>")
    a("<p class='muted'>纵轴为页/秒（一页 = 1 次 layout + 全部版面块），横轴并发文档数按 log₂ 等距。"
      "曲线走平即吞吐饱和，再加并发只会推高延迟。标注的是每条曲线的峰值。</p>")
    a(f"<div class='card'>{legend}"
      f"{facets('fps', y_tick_fmt=lambda t: f'{t:.1f}', label_peak=True)}</div></section>")

    a("<section><h2>单页端到端 p99 延迟随并发的变化</h2>")
    a(f"<p class='muted'>纵轴为秒。红色虚线是 {thresh_s:.0f}s 判定阈值，越线即不达标。</p>")
    a(f"<div class='card'>{legend}"
      f"{facets('p99', y_tick_fmt=lambda t: f'{t:.0f}', threshold=thresh_s)}</div></section>")

    # ---- 明细 (图表的表格孪生) ----
    a("<section><h2>分档明细</h2>")
    a("<p class='muted'>上面两张图的表格版本 —— 每个值都能直接读到，不依赖悬停。</p>")
    for i, t in enumerate(targets):
        a(f"<h3><span class='dot' style='background:var(--series-{i + 1})'></span>"
          f"{esc(t['name'])} · {esc(t['base_url'])} · GPU{t['gpu']} · "
          f"平均 {t['avg_blocks_per_page']:.1f} 块/页</h3>")
        for pr in t["profiles"]:
            a(f"<details><summary>{esc(pr['profile'])} · {pr['pages_in_pool']} 页样本 · "
              f"平均 {pr['avg_blocks_per_page']:.1f} 块/页 · "
              f"建议并发 {pr['recommended_concurrency'] or '—'}</summary>")
            a("<table><thead><tr><th>并发</th><th>吞吐 页/s</th><th>req/s</th>"
              "<th>e2e p50</th><th>e2e p90</th><th>e2e p99</th><th>layout p50</th>"
              "<th>输出 tok/s</th><th>错误率</th><th>前缀缓存</th>"
              "<th>整卡显存</th><th>GPU 利用率</th></tr></thead><tbody>")
            for l in sorted(pr["levels"], key=lambda z: z["concurrency"]):
                gpu = l.get("gpu") or {}
                cls = " class='rec'" if l["concurrency"] == pr["recommended_concurrency"] else ""
                errc = " class='bad'" if l["error_rate"] > report["error_rate_threshold"] else ""
                a(f"<tr{cls}><td>{l['concurrency']}</td><td>{l['fps']:.3f}</td>"
                  f"<td>{l['req_per_s']:.1f}</td><td>{fmt_ms(l['e2e_p50_ms'])}</td>"
                  f"<td>{fmt_ms(l['e2e_p90_ms'])}</td><td>{fmt_ms(l['e2e_p99_ms'])}</td>"
                  f"<td>{fmt_ms(l['layout_p50_ms'])}</td>"
                  f"<td>{l['completion_tokens_per_s']:.0f}</td>"
                  f"<td{errc}>{l['error_rate'] * 100:.1f}%</td>"
                  f"<td>{fmt_hit(l.get('prefix_cache'))}</td>"
                  f"<td>{gpu.get('mem_used_mb_max', 0):.0f} MB</td>"
                  f"<td>{gpu.get('util_pct_avg', 0):.0f}%</td></tr>")
            a("</tbody></table>")
            samples = [s for l in pr["levels"] for s in l.get("error_samples", [])]
            if samples:
                a("<p class='muted'>错误样本：" + "；".join(
                    f"{esc(s['message'][:110])} ×{s['count']}" for s in samples[:4]) + "</p>")
            a("</details>")
    a("</section>")

    # ---- 数据集 ----
    a("<section><h2>压测数据集</h2>")
    a(f"<p class='muted'>从 {ds['catalog_total']} 份真实 PDF 语料按 文件大小 × 页数 分层抽样，"
      f"每档 {ds['docs_per_profile']} 篇 × 每篇 {ds['pages_per_doc']} 页"
      f"（文档内首/中/尾均匀取），渲染 dpi={ds['dpi']}，随机种子 {ds['seed']}，可复现。</p>")
    a("<table><thead><tr><th>分档</th><th>文件</th><th>大小</th><th>总页数</th>"
      "<th>章节数</th><th>取样页</th></tr></thead><tbody>")
    for e in ds["entries"]:
        a(f"<tr><td>{esc(e['profile'])}</td><td>{esc(e['file'][:52])}</td>"
          f"<td>{e['size_bytes'] / 1024 / 1024:.2f} MB</td><td>{e['pages']}</td>"
          f"<td>{e['chapters'] if e['chapters'] else '—'}</td>"
          f"<td>{esc(e['page_indices'])}</td></tr>")
    a("</tbody></table></section>")

    a("<footer>由 <code>bench_ocr_e2e.py</code> 生成。复现："
      "<code>bench_ocr_e2e.py build-dataset</code> → <code>bench_ocr_e2e.py run</code>。"
      "重调阈值无需重跑压测：<code>bench_ocr_e2e.py report --p99-threshold-ms …</code>。</footer>")
    a("</div></div>")
    a("<script id='bench-data' type='application/json'>"
      + json.dumps({"charts": charts}, ensure_ascii=False).replace("</", "<\\/")
      + "</script>")
    a(f"<script>{JS}</script></body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))
