"""
generation.py
Model returns rich JSON → Python builds a detailed slide.
More content, better layout, smarter diagram labels.
"""

import json
import re
import random
import urllib.request

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

JSON_PROMPT = """You are a knowledge extraction assistant. Extract detailed information from the context and return ONLY a valid JSON object. No explanation, no markdown, no code fences.

Return this exact JSON:
{
  "title": "Clear descriptive title (6 words max)",
  "subtitle": "One sentence that defines the concept (20 words max)",
  "topic_badge": "2-3 word category in UPPERCASE",
  "bullets": [
    "Detailed point 1 explaining a key aspect (max 20 words)",
    "Detailed point 2 explaining another aspect (max 20 words)",
    "Detailed point 3 with specific detail or example (max 20 words)",
    "Detailed point 4 with an important implication (max 20 words)"
  ],
  "key_terms": ["term1", "term2", "term3"],
  "diagram_type": "one of: layers | flowchart | cycle | split | network | tree | timeline",
  "diagram_title": "What the diagram shows (4 words max)",
  "nodes": [
    "descriptive node label 1",
    "descriptive node label 2",
    "descriptive node label 3",
    "descriptive node label 4",
    "descriptive node label 5"
  ]
}

RULES:
- bullets: write 4 informative sentences using real facts from the context. Each must be specific and meaningful, not generic.
- subtitle: a precise one-sentence definition.
- key_terms: 3 important technical terms from the context.
- diagram_type: choose based on concept structure:
  * layers → neural networks, deep learning, stacked architecture, encoder-decoder
  * flowchart → training pipeline, algorithm steps, data processing
  * cycle → iterative loops, GAN, diffusion, feedback, RL
  * split → two things compared (e.g. supervised vs unsupervised)
  * network → attention, transformer, graph connections
  * tree → types/taxonomy/classification hierarchy
  * timeline → history, evolution, progression of versions
- nodes: 5 descriptive labels (not single words — use 2-3 words each) that reflect real components or steps from the context.
- Return ONLY the JSON. Nothing before or after it.
"""


def _check_ollama() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False


def _call_ollama(prompt: str, num_predict: int) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": num_predict,
            "seed": random.randint(0, 2**31 - 1)
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )

    parts = []
    count = 0
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts.append(chunk.get("response", ""))
            count += 1
            if count % 40 == 0:
                print(".", end="", flush=True)
            if chunk.get("done"):
                break

    return "".join(parts).strip()


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#f4f6fb",
    "slide":    "#ffffff",
    "border":   "#e2e8f0",
    "badge_bg": "#ede9fe",
    "badge_fg": "#6d28d9",
    "title":    "#0f172a",
    "subtitle": "#475569",
    "bullet":   "#1e293b",
    "bullet2":  "#475569",
    "dim_bg":   "#f0edff",
    "dim_bd":   "#c4b5fd",
    "s1":       "#6d28d9",
    "s2":       "#7c3aed",
    "s3":       "#8b5cf6",
    "s4":       "#a78bfa",
    "s5":       "#ede9fe",
    "sw":       "#ffffff",
    "sd":       "#4c1d95",
    "arrow":    "#6d28d9",
    "tag_bg":   "#ede9fe",
    "tag_fg":   "#6d28d9",
    "footer":   "#94a3b8",
}

ARROW_DEF = f"""<defs>
  <marker id="arr" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <polygon points="0 0, 9 3.5, 0 7" fill="{C['arrow']}"/>
  </marker>
  <marker id="arr2" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
    <polygon points="0 0, 7 2.5, 0 5" fill="{C['s3']}"/>
  </marker>
</defs>"""


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [text]


def _text_block(x, y, text, max_chars, font_size, fill, anchor="middle", weight="600"):
    lines = _wrap(text, max_chars)[:2]
    parts = []
    offset = -(len(lines) - 1) * (font_size * 0.6)
    for i, line in enumerate(lines):
        ty = y + offset + i * (font_size + 3)
        parts.append(
            f'<text x="{x}" y="{ty}" text-anchor="{anchor}" '
            f'font-family="Arial,system-ui" font-size="{font_size}" '
            f'font-weight="{weight}" fill="{fill}">{line}</text>'
        )
    return "\n".join(parts)


# ── Diagram builders ──────────────────────────────────────────────────────────

def _layers(nodes, w, h):
    n = len(nodes)
    bw, bh = min(300, w - 60), 42
    gap = max(10, (h - n * bh) // (n + 1))
    cx = w // 2
    fills = [C["s1"], C["s2"], C["s3"], C["s4"], C["s5"]]
    fgs   = [C["sw"], C["sw"], C["sw"], C["sw"], C["sd"]]
    parts = [ARROW_DEF]
    for i, node in enumerate(nodes):
        y = gap + i * (bh + gap)
        f, fg = fills[i % len(fills)], fgs[i % len(fgs)]
        parts.append(f'<rect x="{cx-bw//2}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{f}" stroke="{C["s3"]}" stroke-width="1"/>')
        parts.append(_text_block(cx, y + bh//2 + 5, node, 32, 13, fg))
        if i < n - 1:
            ay = y + bh + 2
            parts.append(f'<line x1="{cx}" y1="{ay}" x2="{cx}" y2="{ay+gap-6}" stroke="{C["arrow"]}" stroke-width="2" marker-end="url(#arr)"/>')
    return "\n".join(parts)


def _flowchart(nodes, w, h):
    n = len(nodes)
    bw = min(int((w - 40) / n - 20), 130)
    bh = 50
    gap = max(14, (w - 40 - n * bw) // max(n - 1, 1))
    total_w = n * bw + (n - 1) * gap
    sx = (w - total_w) // 2
    cy = h // 2
    fills = [C["s1"], C["s2"], C["s3"], C["s4"], C["s5"]]
    fgs   = [C["sw"], C["sw"], C["sw"], C["sw"], C["sd"]]
    parts = [ARROW_DEF]
    for i, node in enumerate(nodes):
        x = sx + i * (bw + gap)
        f, fg = fills[i % len(fills)], fgs[i % len(fgs)]
        parts.append(f'<rect x="{x}" y="{cy-bh//2}" width="{bw}" height="{bh}" rx="10" fill="{f}" stroke="{C["s3"]}" stroke-width="1"/>')
        parts.append(_text_block(x + bw//2, cy + 5, node, 14, 12, fg))
        if i < n - 1:
            ax = x + bw + 2
            parts.append(f'<line x1="{ax}" y1="{cy}" x2="{ax+gap-6}" y2="{cy}" stroke="{C["arrow"]}" stroke-width="2" marker-end="url(#arr)"/>')
    return "\n".join(parts)


def _cycle(nodes, w, h):
    import math
    n = len(nodes)
    cx, cy = w // 2, h // 2
    rad = min(w, h) // 2 - 55
    bw, bh = 115, 40
    parts = [ARROW_DEF]
    positions = []
    for i in range(n):
        angle = math.radians(-90 + i * 360 / n)
        nx = cx + int(rad * math.cos(angle))
        ny = cy + int(rad * math.sin(angle))
        positions.append((nx, ny))
        f = [C["s1"], C["s2"], C["s3"], C["s4"]][i % 4]
        parts.append(f'<rect x="{nx-bw//2}" y="{ny-bh//2}" width="{bw}" height="{bh}" rx="10" fill="{f}" stroke="{C["s3"]}" stroke-width="1"/>')
        parts.append(_text_block(nx, ny + 5, nodes[i], 14, 11, C["sw"]))
    for i in range(n):
        x1, y1 = positions[i]
        x2, y2 = positions[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        dist = max((dx**2 + dy**2) ** 0.5, 1)
        sx2 = x1 + dx / dist * (bw // 2 + 6)
        sy2 = y1 + dy / dist * (bh // 2 + 6)
        ex2 = x2 - dx / dist * (bw // 2 + 12)
        ey2 = y2 - dy / dist * (bh // 2 + 12)
        parts.append(f'<line x1="{sx2:.0f}" y1="{sy2:.0f}" x2="{ex2:.0f}" y2="{ey2:.0f}" stroke="{C["arrow"]}" stroke-width="2" marker-end="url(#arr)"/>')
    return "\n".join(parts)


def _split(nodes, w, h):
    half = max(len(nodes) // 2, 1)
    left, right = nodes[:half], nodes[half:]
    cx, cy = w // 2, h // 2
    bw, bh, gap = 130, 40, 14
    parts = [ARROW_DEF]
    parts.append(f'<line x1="{cx}" y1="10" x2="{cx}" y2="{h-10}" stroke="{C["border"]}" stroke-width="1.5" stroke-dasharray="6,4"/>')

    def side(nlist, base_x, fill):
        total = len(nlist) * bh + (len(nlist) - 1) * gap
        sy = cy - total // 2
        for i, n in enumerate(nlist):
            y = sy + i * (bh + gap)
            f = C["s1"] if i % 2 == 0 else C["s3"]
            parts.append(f'<rect x="{base_x-bw//2}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{f}" stroke="{C["s3"]}" stroke-width="1"/>')
            parts.append(_text_block(base_x, y + bh//2 + 5, n, 16, 12, C["sw"]))

    side(left,  cx // 2,       C["s1"])
    side(right, cx + cx // 2,  C["s2"])
    # Column labels
    if left:  parts.append(f'<text x="{cx//2}" y="22" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700" fill="{C["badge_fg"]}" letter-spacing="1">{left[0].split()[0].upper() if left else "A"}</text>')
    if right: parts.append(f'<text x="{cx+cx//2}" y="22" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700" fill="{C["badge_fg"]}" letter-spacing="1">{right[0].split()[0].upper() if right else "B"}</text>')
    return "\n".join(parts)


def _network(nodes, w, h):
    import math
    n = len(nodes)
    cx, cy = w // 2, h // 2
    r_node = 36
    r_orbit = min(w, h) // 2 - 44
    parts = [ARROW_DEF]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_node+4}" fill="{C["s1"]}" stroke="{C["s2"]}" stroke-width="2"/>')
    parts.append(_text_block(cx, cy + 5, nodes[0], 10, 12, C["sw"]))
    for i, node in enumerate(nodes[1:], 1):
        angle = math.radians(-90 + (i - 1) * 360 / max(n - 1, 1))
        nx = cx + int(r_orbit * math.cos(angle))
        ny = cy + int(r_orbit * math.sin(angle))
        sx2 = cx + int((r_node + 8) * math.cos(angle))
        sy2 = cy + int((r_node + 8) * math.sin(angle))
        ex2 = nx - int((r_node + 14) * math.cos(angle))
        ey2 = ny - int((r_node + 14) * math.sin(angle))
        parts.append(f'<line x1="{sx2}" y1="{sy2}" x2="{ex2}" y2="{ey2}" stroke="{C["arrow"]}" stroke-width="1.5" marker-end="url(#arr)"/>')
        f = [C["s2"], C["s3"], C["s4"], C["s1"]][i % 4]
        parts.append(f'<circle cx="{nx}" cy="{ny}" r="{r_node}" fill="{f}" stroke="{C["s3"]}" stroke-width="1.5"/>')
        parts.append(_text_block(nx, ny + 5, node, 10, 11, C["sw"]))
    return "\n".join(parts)


def _tree(nodes, w, h):
    bw, bh = 130, 40
    root = nodes[0]
    children = nodes[1:]
    n = len(children)
    cx = w // 2
    root_y = 20
    child_y = root_y + bh + 60
    parts = [ARROW_DEF]
    parts.append(f'<rect x="{cx-bw//2}" y="{root_y}" width="{bw}" height="{bh}" rx="10" fill="{C["s1"]}"/>')
    parts.append(_text_block(cx, root_y + bh//2 + 5, root, 16, 13, C["sw"]))
    if not children: return "\n".join(parts)
    total = n * bw + (n - 1) * 16
    sx2 = cx - total // 2
    # Horizontal connector
    parts.append(f'<line x1="{cx}" y1="{root_y+bh}" x2="{cx}" y2="{child_y-16}" stroke="{C["s3"]}" stroke-width="1.5"/>')
    if n > 1:
        lx = sx2 + bw // 2
        rx = sx2 + (n-1) * (bw + 16) + bw // 2
        parts.append(f'<line x1="{lx}" y1="{child_y-16}" x2="{rx}" y2="{child_y-16}" stroke="{C["s3"]}" stroke-width="1.5"/>')
    for i, ch in enumerate(children):
        cx2 = sx2 + i * (bw + 16) + bw // 2
        parts.append(f'<line x1="{cx2}" y1="{child_y-16}" x2="{cx2}" y2="{child_y}" stroke="{C["s3"]}" stroke-width="1.5" marker-end="url(#arr)"/>')
        f = C["s2"] if i % 2 == 0 else C["s3"]
        parts.append(f'<rect x="{cx2-bw//2}" y="{child_y}" width="{bw}" height="{bh}" rx="10" fill="{f}"/>')
        parts.append(_text_block(cx2, child_y + bh//2 + 5, ch, 16, 12, C["sw"]))
    return "\n".join(parts)


def _timeline(nodes, w, h):
    n = len(nodes)
    cy = h // 2
    mg = 50
    step = (w - 2 * mg) // max(n - 1, 1)
    bw, bh = 100, 36
    parts = [ARROW_DEF]
    parts.append(f'<line x1="{mg}" y1="{cy}" x2="{w-mg-10}" y2="{cy}" stroke="{C["s1"]}" stroke-width="2.5" marker-end="url(#arr)"/>')
    for i, node in enumerate(nodes):
        x = mg + i * step
        above = i % 2 == 0
        by = (cy - bh - 28) if above else (cy + 28)
        sy2 = cy - 10 if above else cy + 10
        ey2 = by + bh if above else by
        parts.append(f'<circle cx="{x}" cy="{cy}" r="8" fill="{C["s1"]}" stroke="{C["s3"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{x}" y1="{sy2}" x2="{x}" y2="{ey2}" stroke="{C["s3"]}" stroke-width="1.5"/>')
        f = C["s1"] if i % 2 == 0 else C["s2"]
        parts.append(f'<rect x="{x-bw//2}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{f}"/>')
        parts.append(_text_block(x, by + bh//2 + 5, node, 13, 11, C["sw"]))
    return "\n".join(parts)


BUILDERS = {
    "layers":    _layers,
    "flowchart": _flowchart,
    "cycle":     _cycle,
    "split":     _split,
    "network":   _network,
    "tree":      _tree,
    "timeline":  _timeline,
}


def build_slide(data: dict) -> str:
    title      = data.get("title", "Concept Overview")
    subtitle   = data.get("subtitle", "")
    badge      = data.get("topic_badge", "TOPIC")
    bullets    = data.get("bullets", [])[:4]
    key_terms  = data.get("key_terms", [])[:3]
    diag_type  = data.get("diagram_type", "flowchart")
    diag_title = data.get("diagram_title", "Diagram")
    nodes      = data.get("nodes", ["A", "B", "C", "D", "E"])[:6]

    builder  = BUILDERS.get(diag_type, _flowchart)
    svg_w, svg_h = 530, 240
    svg_inner = builder(nodes, svg_w, svg_h)

    bullet_html = "\n".join(
        f'<li>{b}</li>' for b in bullets
    )

    tags_html = " ".join(
        f'<span class="tag">{t}</span>' for t in key_terms
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  width: 960px; height: 640px; overflow: hidden;
  background: {C["bg"]};
  font-family: Arial, system-ui, -apple-system, sans-serif;
  display: flex; align-items: center; justify-content: center;
}}
.slide {{
  width: 920px; height: 608px;
  background: {C["slide"]};
  border-radius: 18px;
  box-shadow: 0 2px 24px rgba(109,40,217,0.10), 0 1px 4px rgba(0,0,0,0.06);
  display: grid;
  grid-template-rows: auto 1px 1fr auto;
  overflow: hidden;
}}
/* ── Header ── */
.header {{
  padding: 22px 28px 16px;
  display: flex; flex-direction: column; gap: 6px;
}}
.badge {{
  display: inline-flex; align-items: center; gap: 6px;
  background: {C["badge_bg"]}; color: {C["badge_fg"]};
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  border-radius: 20px; padding: 3px 12px; width: fit-content;
  text-transform: uppercase;
}}
.badge::before {{ content: "●"; font-size: 8px; }}
.title {{ font-size: 28px; font-weight: 800; color: {C["title"]}; line-height: 1.15; letter-spacing: -0.4px; }}
.subtitle {{ font-size: 13px; color: {C["subtitle"]}; line-height: 1.5; max-width: 600px; }}
/* ── Divider ── */
.hr {{ background: {C["border"]}; }}
/* ── Body ── */
.body {{
  display: grid; grid-template-columns: 1fr 1.15fr;
  gap: 0; padding: 0; overflow: hidden;
}}
/* Left: bullets */
.left {{
  padding: 18px 20px 18px 28px;
  display: flex; flex-direction: column; gap: 10px;
  border-right: 1px solid {C["border"]};
  overflow: hidden;
}}
.bullets {{ list-style: none; display: flex; flex-direction: column; gap: 0; }}
.bullets li {{
  font-size: 13px; color: {C["bullet"]}; line-height: 1.55;
  padding: 7px 0 7px 18px; position: relative;
  border-bottom: 1px solid {C["border"]};
}}
.bullets li:last-child {{ border-bottom: none; }}
.bullets li::before {{
  content: ""; position: absolute; left: 0; top: 50%;
  transform: translateY(-50%);
  width: 7px; height: 7px; border-radius: 50%;
  background: {C["s1"]};
}}
.tags {{ display: flex; gap: 6px; flex-wrap: wrap; padding-top: 4px; }}
.tag {{
  background: {C["tag_bg"]}; color: {C["tag_fg"]};
  font-size: 10px; font-weight: 600; border-radius: 12px;
  padding: 2px 10px; letter-spacing: 0.04em;
}}
/* Right: diagram */
.right {{
  padding: 16px 20px 16px 18px;
  display: flex; flex-direction: column; gap: 8px;
  background: {C["bg"]};
}}
.diag-label {{
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  color: {C["badge_fg"]}; text-transform: uppercase;
}}
.diag-box {{
  background: {C["dim_bg"]}; border-radius: 12px;
  border: 1px solid {C["dim_bd"]};
  flex: 1; display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}}
/* ── Footer ── */
.footer {{
  padding: 8px 28px;
  background: {C["bg"]};
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid {C["border"]};
}}
.footer span {{ font-size: 10px; color: {C["footer"]}; }}
.footer .model {{
  background: {C["badge_bg"]}; color: {C["badge_fg"]};
  padding: 2px 10px; border-radius: 10px;
  font-size: 10px; font-weight: 600;
}}
</style>
</head>
<body>
<div class="slide">
  <div class="header">
    <div class="badge">{badge}</div>
    <div class="title">{title}</div>
    {'<div class="subtitle">' + subtitle + '</div>' if subtitle else ''}
  </div>
  <div class="hr"></div>
  <div class="body">
    <div class="left">
      <ul class="bullets">{bullet_html}</ul>
      {'<div class="tags">' + tags_html + '</div>' if tags_html else ''}
    </div>
    <div class="right">
      <div class="diag-label">▸ {diag_title}</div>
      <div class="diag-box">
        <svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">
          <rect width="{svg_w}" height="{svg_h}" fill="transparent"/>
          {svg_inner}
        </svg>
      </div>
    </div>
  </div>
  <div class="footer">
    <span>Generated from book context · Hybrid RAG</span>
    <span class="model">{OLLAMA_MODEL}</span>
  </div>
</div>
</body>
</html>"""


class AnswerGenerator:

    def __init__(self):
        if not _check_ollama():
            raise RuntimeError(
                "Ollama is not running.\n"
                "Install from https://ollama.com/download\n"
                "Then run: ollama pull qwen2.5:3b"
            )
        print(f"[generation] Ollama ready — {OLLAMA_MODEL}")

    def generate(self, question: str, chunks: list[str]) -> str:
        # Use top 5 chunks for richer context
        context = "\n\n".join(chunks[:3])
        prompt = (
            f"{JSON_PROMPT}\n\n"
            f"Question: {question}\n\n"
            f"Context from book:\n{context}\n\n"
            "Return ONLY the JSON:"
        )
        print("[generation] Extracting content", end="", flush=True)
        raw = _call_ollama(prompt, num_predict=400)
        print(" ✓")

        try:
            data = _parse_json(raw)
        except Exception as e:
            print(f"[generation] JSON parse error: {e} | raw: {raw[:200]}")
            data = {
                "title": question[:50],
                "subtitle": "Based on the uploaded book context.",
                "topic_badge": "OVERVIEW",
                "bullets": [
                    "The context provides relevant information about this topic.",
                    "Key concepts are explained throughout the referenced chapters.",
                    "Multiple aspects and definitions are covered in the source material.",
                    "Try rephrasing your question for a more specific answer.",
                ],
                "key_terms": ["concept", "context", "reference"],
                "diagram_type": "flowchart",
                "diagram_title": "General Overview",
                "nodes": ["Input Data", "Processing", "Analysis", "Output", "Result"],
            }

        print("[generation] Building slide ✓")
        return build_slide(data)
