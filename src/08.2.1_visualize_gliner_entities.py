#!/usr/bin/env python3
"""
08.2.1 / GLiNER entity-validation visualization.

Reads the GLiNER extraction JSON and creates a self-contained interactive HTML
report where detected entities are highlighted directly inside the original
chunk text.

This is meant to validate:
  1. entity boundaries,
  2. entity labels/types,
  3. confidence scores,
  4. coverage across documents.

Usage:
    python 08.3_visualize_gliner_entities.py \
        --input graphResults/entity_extraction/gliner_entities.json \
        --output graphResults/entity_extraction/gliner_entity_validation.html

Optional:
    --min-confidence 0.50
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Create interactive HTML visualization of GLiNER entities."
    )
    p.add_argument("--input", required=True, help="Path to gliner_entities.json")
    p.add_argument(
        "--output",
        default="graphResults/entity_extraction/gliner_entity_validation.html",
        help="Output HTML file.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only include entities at or above this GLiNER confidence.",
    )
    return p.parse_args()


def get_chunks(data):
    if isinstance(data, list):
        return data

    for key in ("chunks", "results", "documents"):
        value = data.get(key)
        if isinstance(value, list):
            return value

    raise ValueError(
        "Could not find chunk list. Expected a list or one of "
        "the keys: chunks, results, documents."
    )


def get_text(chunk):
    for key in ("text", "chunk_text", "content"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("Chunk is missing its text.")


def get_entities(chunk):
    for key in ("entities", "predictions"):
        value = chunk.get(key)
        if isinstance(value, list):
            return value
    return []


def get_source(chunk):
    source = chunk.get("source")
    if source:
        return str(source)

    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        source = metadata.get("source")
        if source:
            return str(source)

    return "unknown"


def get_chunk_id(chunk, index):
    return str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{index:03d}")


def entity_type(entity):
    return str(
        entity.get("entity_type")
        or entity.get("label")
        or entity.get("type")
        or "ENTITY"
    )


def render_annotated_text(text, entities):
    """
    Render non-overlapping entity spans.

    For overlapping spans, keep the higher-confidence span first.
    This is only for visualization; the raw JSON is not modified.
    """
    candidates = []

    for e in entities:
        try:
            start = int(e["start"])
            end = int(e["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (0 <= start < end <= len(text)):
            continue

        score = float(e.get("score", 0.0) or 0.0)

        candidates.append({
            "start": start,
            "end": end,
            "text": str(e.get("text", text[start:end])),
            "type": entity_type(e),
            "score": score,
        })

    # Higher confidence gets priority when spans overlap.
    priority = sorted(
        candidates,
        key=lambda x: (-x["score"], x["start"], -(x["end"] - x["start"]))
    )

    accepted = []

    for e in priority:
        overlaps = any(
            not (e["end"] <= a["start"] or e["start"] >= a["end"])
            for a in accepted
        )
        if not overlaps:
            accepted.append(e)

    accepted.sort(key=lambda x: x["start"])

    output = []
    cursor = 0

    for e in accepted:
        output.append(html.escape(text[cursor:e["start"]]))

        etype = html.escape(e["type"])
        score = e["score"]
        mention = html.escape(text[e["start"]:e["end"]])

        output.append(
            f'<mark class="entity" '
            f'data-type="{etype}" '
            f'data-score="{score:.6f}" '
            f'title="{etype} • confidence {score:.3f}">'
            f'<span class="mention">{mention}</span>'
            f'<span class="entity-tag">{etype} · {score:.2f}</span>'
            f'</mark>'
        )

        cursor = e["end"]

    output.append(html.escape(text[cursor:]))
    return "".join(output), accepted


def main():
    args = parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = get_chunks(data)

    rendered_chunks = []
    type_counts = Counter()
    source_counts = Counter()
    scores = []
    total_raw = 0

    for i, chunk in enumerate(chunks):
        text = get_text(chunk)
        source = get_source(chunk)
        cid = get_chunk_id(chunk, i)

        entities = []
        for e in get_entities(chunk):
            total_raw += 1
            try:
                score = float(e.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0

            if score >= args.min_confidence:
                entities.append(e)

        annotated, accepted = render_annotated_text(text, entities)

        for e in accepted:
            type_counts[e["type"]] += 1
            scores.append(e["score"])

        source_counts[source] += len(accepted)

        rendered_chunks.append({
            "chunk_id": cid,
            "source": source,
            "entity_count": len(accepted),
            "html": annotated,
        })

    types = sorted(type_counts)
    sources = sorted(source_counts)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    type_options = "\n".join(
        f'<option value="{html.escape(t)}">{html.escape(t)} ({type_counts[t]})</option>'
        for t in types
    )

    source_options = "\n".join(
        f'<option value="{html.escape(s)}">{html.escape(Path(s).name)} ({source_counts[s]})</option>'
        for s in sources
    )

    type_rows = "\n".join(
        f"""
        <div class="type-row">
          <span class="type-name">{html.escape(t)}</span>
          <span class="bar-wrap">
            <span class="bar" style="width:{(type_counts[t] / max(type_counts.values(), default=1)) * 100:.1f}%"></span>
          </span>
          <span class="count">{type_counts[t]}</span>
        </div>
        """
        for t in sorted(type_counts, key=lambda x: (-type_counts[x], x))
    )

    chunk_cards = "\n".join(
        f"""
        <section class="chunk-card"
                 data-source="{html.escape(c['source'])}"
                 data-entity-count="{c['entity_count']}">
          <div class="chunk-meta">
            <div>
              <strong>{html.escape(c['chunk_id'])}</strong>
              <span class="source">{html.escape(Path(c['source']).name)}</span>
            </div>
            <span class="badge">{c['entity_count']} entities</span>
          </div>
          <div class="chunk-text">{c['html']}</div>
        </section>
        """
        for c in rendered_chunks
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLiNER Entity Validation</title>
<style>
  :root {{
    --bg: #f6f7f9;
    --panel: #ffffff;
    --text: #18212f;
    --muted: #667085;
    --border: #d9dee7;
    --accent: #315efb;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }}

  .page {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 28px;
  }}

  h1 {{
    margin: 0 0 6px;
    font-size: 28px;
  }}

  .subtitle {{
    color: var(--muted);
    margin-bottom: 22px;
  }}

  .stats {{
    display: grid;
    grid-template-columns: repeat(5, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }}

  .stat {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
  }}

  .stat .label {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 4px;
  }}

  .stat .value {{
    font-size: 21px;
    font-weight: 700;
  }}

  .layout {{
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    gap: 18px;
  }}

  .sidebar {{
    min-width: 0;
  }}

  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
  }}

  .panel h2 {{
    font-size: 15px;
    margin: 0 0 12px;
  }}

  label {{
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin: 12px 0 5px;
  }}

  select, input[type="range"] {{
    width: 100%;
  }}

  select {{
    padding: 9px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: white;
  }}

  .threshold-row {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  #thresholdValue {{
    width: 42px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}

  .type-row {{
    display: grid;
    grid-template-columns: 115px 1fr 30px;
    gap: 8px;
    align-items: center;
    margin: 7px 0;
    font-size: 11px;
  }}

  .type-name {{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .bar-wrap {{
    height: 7px;
    background: #eceff3;
    border-radius: 999px;
    overflow: hidden;
  }}

  .bar {{
    display: block;
    height: 100%;
    background: var(--accent);
  }}

  .count {{
    text-align: right;
    color: var(--muted);
  }}

  .main {{
    min-width: 0;
  }}

  .results-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 9px;
  }}

  #visibleCount {{
    color: var(--muted);
    font-size: 13px;
  }}

  .chunk-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
  }}

  .chunk-meta {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 11px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }}

  .source {{
    color: var(--muted);
    margin-left: 8px;
  }}

  .badge {{
    color: var(--muted);
  }}

  .chunk-text {{
    padding: 18px;
    line-height: 2.15;
    white-space: pre-wrap;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 16px;
  }}

  mark.entity {{
    --hue: 210;
    position: relative;
    display: inline;
    background: hsl(var(--hue) 85% 88%);
    color: inherit;
    border-bottom: 2px solid hsl(var(--hue) 55% 48%);
    border-radius: 3px;
    padding: 1px 2px;
    margin: 0 1px;
  }}

  .entity-tag {{
    display: inline-block;
    margin-left: 4px;
    padding: 1px 4px;
    border-radius: 4px;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 9px;
    font-weight: 700;
    line-height: 1.4;
    vertical-align: middle;
    background: hsl(var(--hue) 55% 48%);
    color: white;
    white-space: nowrap;
  }}

  mark.entity.filtered-out {{
    background: transparent;
    border-bottom-color: transparent;
  }}

  mark.entity.filtered-out .entity-tag {{
    display: none;
  }}

  .hidden {{
    display: none !important;
  }}

  .note {{
    color: var(--muted);
    font-size: 11px;
    line-height: 1.45;
  }}

  @media (max-width: 850px) {{
    .page {{ padding: 14px; }}
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    .layout {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="page">
  <h1>GLiNER Entity Validation</h1>
  <div class="subtitle">
    Inspect entity boundaries, entity types, confidence and document coverage directly in source text.
  </div>

  <div class="stats">
    <div class="stat">
      <div class="label">Chunks</div>
      <div class="value">{len(rendered_chunks)}</div>
    </div>
    <div class="stat">
      <div class="label">Displayed entities</div>
      <div class="value">{len(scores)}</div>
    </div>
    <div class="stat">
      <div class="label">Entity types</div>
      <div class="value">{len(types)}</div>
    </div>
    <div class="stat">
      <div class="label">Mean confidence</div>
      <div class="value">{avg_score:.3f}</div>
    </div>
    <div class="stat">
      <div class="label">Confidence range</div>
      <div class="value">{min_score:.2f}–{max_score:.2f}</div>
    </div>
  </div>

  <div class="layout">
    <aside class="sidebar">
      <div class="panel">
        <h2>Filters</h2>

        <label for="sourceFilter">Source document</label>
        <select id="sourceFilter">
          <option value="">All documents</option>
          {source_options}
        </select>

        <label for="typeFilter">Entity type</label>
        <select id="typeFilter">
          <option value="">All entity types</option>
          {type_options}
        </select>

        <label for="threshold">Minimum confidence</label>
        <div class="threshold-row">
          <input id="threshold" type="range" min="0" max="1" step="0.01"
                 value="{max(0.0, min(1.0, args.min_confidence)):.2f}">
          <span id="thresholdValue">{max(0.0, min(1.0, args.min_confidence)):.2f}</span>
        </div>
      </div>

      <div class="panel">
        <h2>Entity-type distribution</h2>
        {type_rows}
      </div>

      <div class="panel note">
        Validation tip: inspect three things independently:
        <strong>span boundary</strong>, <strong>entity type</strong>, and
        <strong>confidence</strong>. A relation model cannot recover an entity
        that was missed or mislabeled here.
      </div>
    </aside>

    <main class="main">
      <div class="results-head">
        <strong>Annotated chunks</strong>
        <span id="visibleCount"></span>
      </div>
      <div id="chunks">
        {chunk_cards}
      </div>
    </main>
  </div>
</div>

<script>
(function () {{
  const sourceFilter = document.getElementById("sourceFilter");
  const typeFilter = document.getElementById("typeFilter");
  const threshold = document.getElementById("threshold");
  const thresholdValue = document.getElementById("thresholdValue");
  const visibleCount = document.getElementById("visibleCount");

  const cards = [...document.querySelectorAll(".chunk-card")];
  const allEntities = [...document.querySelectorAll("mark.entity")];

  // Stable deterministic color from entity type.
  function hueForType(type) {{
    let hash = 0;
    for (let i = 0; i < type.length; i++) {{
      hash = ((hash << 5) - hash) + type.charCodeAt(i);
      hash |= 0;
    }}
    return Math.abs(hash) % 360;
  }}

  allEntities.forEach(el => {{
    el.style.setProperty("--hue", hueForType(el.dataset.type));
  }});

  function update() {{
    const source = sourceFilter.value;
    const type = typeFilter.value;
    const minScore = Number(threshold.value);

    thresholdValue.textContent = minScore.toFixed(2);

    let shownCards = 0;
    let shownEntities = 0;

    cards.forEach(card => {{
      const sourceMatches = !source || card.dataset.source === source;
      const entities = [...card.querySelectorAll("mark.entity")];

      let matchingEntities = 0;

      entities.forEach(el => {{
        const scoreMatches = Number(el.dataset.score) >= minScore;
        const typeMatches = !type || el.dataset.type === type;
        const visible = scoreMatches && typeMatches;

        el.classList.toggle("filtered-out", !visible);

        if (visible) matchingEntities++;
      }});

      const cardVisible = sourceMatches && matchingEntities > 0;
      card.classList.toggle("hidden", !cardVisible);

      if (cardVisible) {{
        shownCards++;
        shownEntities += matchingEntities;
      }}
    }});

    visibleCount.textContent =
      `${{shownEntities}} entities in ${{shownCards}} chunks`;
  }}

  sourceFilter.addEventListener("change", update);
  typeFilter.addEventListener("change", update);
  threshold.addEventListener("input", update);

  update();
}})();
</script>
</body>
</html>
"""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")

    print(f"Input chunks: {len(chunks)}")
    print(f"Raw entity predictions: {total_raw}")
    print(f"Displayed entity mentions: {len(scores)}")
    print(f"Entity types: {len(types)}")
    print(f"Mean confidence: {avg_score:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
