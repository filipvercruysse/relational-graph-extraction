"""
Benchmark GLiNER2 (and an LLM) against ground-truth entity + relation extraction.

Usage
-----
# Step 1 – Run GLiNER2 and score against ground truth:
  python benchmark_ground_truth.py

# Step 2 – After you have an LLM extraction file (llm_extraction.json):
  python benchmark_ground_truth.py --llm-file output/llm_extraction.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# 1. Minimal test text with unambiguous causal structure
# ---------------------------------------------------------------------------

TEST_TEXT = """\
Deforestation removes large areas of trees from tropical rainforests. \
Without tree cover, soil is directly exposed to heavy rainfall, which \
causes erosion. Erosion washes nutrient-rich topsoil into rivers, \
degrading water quality and causing sedimentation. Sedimentation blocks \
river channels, increasing the risk of flooding downstream. Meanwhile, \
the loss of trees reduces carbon absorption, accelerating the rise of \
atmospheric CO2. Higher CO2 levels intensify the greenhouse effect, \
raising global temperatures. Rising temperatures alter precipitation \
patterns, leading to more frequent droughts. Droughts weaken the \
remaining forest, making it more vulnerable to wildfires. Wildfires \
destroy yet more trees, feeding back into deforestation.\
"""

# ---------------------------------------------------------------------------
# 2. Ground-truth graph
# ---------------------------------------------------------------------------

GT_ENTITIES: list[dict[str, str]] = [
    {"text": "deforestation",      "label": "process"},
    {"text": "trees",              "label": "entity"},
    {"text": "soil",               "label": "entity"},
    {"text": "erosion",            "label": "process"},
    {"text": "topsoil",            "label": "entity"},
    {"text": "rivers",             "label": "entity"},
    {"text": "water quality",      "label": "property"},
    {"text": "sedimentation",      "label": "process"},
    {"text": "flooding",           "label": "process"},
    {"text": "carbon absorption",  "label": "process"},
    {"text": "atmospheric CO2",    "label": "entity"},
    {"text": "greenhouse effect",  "label": "process"},
    {"text": "global temperatures","label": "property"},
    {"text": "precipitation patterns", "label": "property"},
    {"text": "droughts",           "label": "process"},
    {"text": "wildfires",          "label": "process"},
]

GT_RELATIONS: list[dict[str, str]] = [
    {"head": "deforestation",      "relation": "removes",       "tail": "trees"},
    {"head": "deforestation",      "relation": "exposes",       "tail": "soil"},
    {"head": "soil",               "relation": "causes",        "tail": "erosion"},
    {"head": "erosion",            "relation": "washes_away",   "tail": "topsoil"},
    {"head": "erosion",            "relation": "degrades",      "tail": "water quality"},
    {"head": "erosion",            "relation": "causes",        "tail": "sedimentation"},
    {"head": "sedimentation",      "relation": "increases",     "tail": "flooding"},
    {"head": "deforestation",      "relation": "reduces",       "tail": "carbon absorption"},
    {"head": "carbon absorption",  "relation": "increases",     "tail": "atmospheric CO2"},
    {"head": "atmospheric CO2",    "relation": "intensifies",   "tail": "greenhouse effect"},
    {"head": "greenhouse effect",  "relation": "raises",        "tail": "global temperatures"},
    {"head": "global temperatures","relation": "alters",        "tail": "precipitation patterns"},
    {"head": "precipitation patterns","relation": "causes",     "tail": "droughts"},
    {"head": "droughts",           "relation": "causes",        "tail": "wildfires"},
    {"head": "wildfires",          "relation": "causes",        "tail": "deforestation"},
]

# Entity and relation schemas matching the ground truth
ENTITY_SCHEMA = {
    "process":  "A natural process, event, or phenomenon",
    "entity":   "A physical thing, substance, or object",
    "property": "A measurable quality, condition, or pattern",
}

RELATION_SCHEMA = {
    "causes":       "The first entity directly causes or produces the second",
    "removes":      "The first entity removes or eliminates the second",
    "exposes":      "The first entity exposes the second to harm",
    "washes_away":  "The first entity washes away or displaces the second",
    "degrades":     "The first entity reduces the quality of the second",
    "increases":    "The first entity increases the magnitude of the second",
    "reduces":      "The first entity decreases the magnitude of the second",
    "intensifies":  "The first entity strengthens or amplifies the second",
    "raises":       "The first entity makes the second higher",
    "alters":       "The first entity changes the pattern of the second",
}


# ---------------------------------------------------------------------------
# 3. GLiNER2 extraction
# ---------------------------------------------------------------------------

def run_gliner2(text: str) -> dict[str, Any]:
    """Run GLiNER2 extraction and return raw result dict."""
    from gliner2 import GLiNER2

    model_name = "fastino/gliner2-base-v1"
    print(f"Loading {model_name} …")
    t0 = time.perf_counter()
    extractor = GLiNER2.from_pretrained(model_name)
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s")

    schema = (
        extractor.create_schema()
        .entities(ENTITY_SCHEMA)
        .relations(RELATION_SCHEMA)
    )

    print("Running extraction …")
    t0 = time.perf_counter()
    result = extractor.extract(
        text, schema,
        include_confidence=True,
        include_spans=True,
    )
    infer_s = time.perf_counter() - t0
    print(f"  done in {infer_s:.1f}s")

    return {
        "model": model_name,
        "load_seconds": round(load_s, 3),
        "inference_seconds": round(infer_s, 3),
        "result": result,
    }


# ---------------------------------------------------------------------------
# 4. Normalisation helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for matching."""
    import re
    return re.sub(r"\s+", " ", text.lower().strip())


def entity_set(entities: list[dict]) -> set[str]:
    """Return a set of normalised entity texts (ignoring labels)."""
    return {normalise(e["text"]) for e in entities}


def relation_triples(relations: list[dict]) -> set[tuple[str, str]]:
    """Return a set of (head, tail) pairs (ignoring relation label)."""
    return {(normalise(r["head"]), normalise(r["tail"])) for r in relations}


def relation_triples_labeled(relations: list[dict]) -> set[tuple[str, str, str]]:
    """Return (head, relation, tail) triples."""
    return {(normalise(r["head"]), normalise(r.get("relation", "")), normalise(r["tail"])) for r in relations}


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    tp_items: list = field(default_factory=list)
    fp_items: list = field(default_factory=list)
    fn_items: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {
            "precision": round(self.precision, 3),
            "recall":    round(self.recall, 3),
            "f1":        round(self.f1, 3),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tp_items": sorted(self.tp_items),
            "fp_items": sorted(self.fp_items),
            "fn_items": sorted(self.fn_items),
        }


def score_entities(gt: list[dict], pred: list[dict], name: str) -> Score:
    """Fuzzy entity matching: GT entity is 'found' if any predicted entity
    text contains it OR it contains the predicted entity text."""
    gt_set = entity_set(gt)
    pred_set = entity_set(pred)

    s = Score(name)
    for g in gt_set:
        if any(g in p or p in g for p in pred_set):
            s.true_positives += 1
            s.tp_items.append(g)
        else:
            s.false_negatives += 1
            s.fn_items.append(g)
    for p in pred_set:
        if not any(g in p or p in g for g in gt_set):
            s.false_positives += 1
            s.fp_items.append(p)
    return s


def score_relations(gt: list[dict], pred: list[dict], name: str) -> Score:
    """Match relations by (head, tail) with fuzzy entity matching.
    A predicted relation matches if both head and tail fuzzy-match GT."""
    gt_pairs = relation_triples(gt)
    pred_pairs = relation_triples(pred)

    def fuzzy_match_pair(gp, pp):
        h_match = gp[0] in pp[0] or pp[0] in gp[0]
        t_match = gp[1] in pp[1] or pp[1] in gp[1]
        return h_match and t_match

    s = Score(name)
    for g in gt_pairs:
        if any(fuzzy_match_pair(g, p) for p in pred_pairs):
            s.true_positives += 1
            s.tp_items.append(f"{g[0]} → {g[1]}")
        else:
            s.false_negatives += 1
            s.fn_items.append(f"{g[0]} → {g[1]}")
    for p in pred_pairs:
        if not any(fuzzy_match_pair(g, p) for g in gt_pairs):
            s.false_positives += 1
            s.fp_items.append(f"{p[0]} → {p[1]}")
    return s


# ---------------------------------------------------------------------------
# 6. Graph visualisation
# ---------------------------------------------------------------------------

def build_nx_graph(entities: list[dict], relations: list[dict], title: str) -> nx.DiGraph:
    G = nx.DiGraph(title=title)
    for e in entities:
        G.add_node(normalise(e["text"]), label=e.get("label", ""))
    for r in relations:
        G.add_edge(
            normalise(r["head"]),
            normalise(r["tail"]),
            label=r.get("relation", ""),
        )
    return G


LABEL_COLORS = {
    "process":  "#e74c3c",
    "entity":   "#3498db",
    "property": "#2ecc71",
    "concept":  "#9b59b6",
    "method":   "#e67e22",
    "model":    "#1abc9c",
}


LABEL_COLORS = {
    "process":  "#E07A5F",   # terracotta
    "entity":   "#3D405B",   # charcoal blue
    "property": "#81B29A",   # sage green
    "concept":  "#F2CC8F",   # warm sand
    "method":   "#F4A261",   # sandy orange
    "model":    "#264653",   # deep teal
}


def _draw_one_graph(G: nx.DiGraph, ax, title: str, pos=None, show_legend=False):
    """Draw a single graph onto an axes. Returns the layout used."""
    import matplotlib.patches as mpatches

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "(empty graph)", ha="center", va="center",
                fontsize=20, transform=ax.transAxes)
        ax.set_title(title, fontsize=24, fontweight="bold", pad=24)
        ax.axis("off")
        return {}

    if pos is None:
        pos = nx.spring_layout(G, k=2.8, iterations=100, seed=42)

    draw_pos = {n: pos[n] for n in G.nodes if n in pos}
    nodelist = [n for n in G.nodes if n in draw_pos]
    node_colors = [LABEL_COLORS.get(G.nodes[n].get("label", "").lower(), "#bbb")
                   for n in nodelist]

    nx.draw_networkx_nodes(G, draw_pos, nodelist=nodelist, ax=ax, node_size=3600,
                           node_color=node_colors, alpha=0.92,
                           edgecolors="white", linewidths=2.5)

    # Node labels ABOVE the circles
    label_pos = {n: (xy[0], xy[1] + 0.085) for n, xy in draw_pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=15, font_weight="bold",
                            font_color="#1a1a1a")

    # Edges
    edgelist = [(u, v) for u, v in G.edges if u in draw_pos and v in draw_pos]
    nx.draw_networkx_edges(G, draw_pos, edgelist=edgelist, ax=ax,
                           arrows=True, arrowsize=24,
                           edge_color="#999999", width=2.0, alpha=0.6,
                           connectionstyle="arc3,rad=0.10",
                           min_source_margin=24, min_target_margin=24)

    # Edge labels — large and bold
    edge_labels = {(u, v): d.get("label", "")
                   for u, v, d in G.edges(data=True)
                   if u in draw_pos and v in draw_pos}
    edge_labels = {k: v for k, v in edge_labels.items() if v}
    if edge_labels:
        nx.draw_networkx_edge_labels(G, draw_pos, edge_labels=edge_labels,
                                     ax=ax, font_size=13, font_color="#2c3e50",
                                     font_weight="bold",
                                     bbox=dict(boxstyle="round,pad=0.12",
                                               fc="white", ec="none", alpha=0.75))

    # Single legend only on the first panel
    if show_legend:
        all_labels = sorted({G.nodes[n].get("label", "").lower()
                             for n in G.nodes} - {""})
        patches = [mpatches.Patch(color=LABEL_COLORS.get(l, "#bbb"), label=l)
                   for l in all_labels]
        if patches:
            ax.legend(handles=patches, loc="upper left", fontsize=14,
                      frameon=False, handlelength=1.5, handleheight=1.5)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=24)
    ax.axis("off")
    return pos


def _compute_anchored_positions(gt_graph: nx.DiGraph,
                                other_graphs: list[nx.DiGraph]) -> dict:
    """Compute positions anchored to the ground truth layout.

    1. Layout the ground truth graph.
    2. For other graphs, reuse GT positions for shared nodes.
    3. Place new (non-GT) nodes in open space around the existing layout.
    """
    import numpy as np

    gt_pos = nx.spring_layout(gt_graph, k=3.5, iterations=150, seed=42)

    # Collect all extra nodes across all graphs
    gt_nodes = set(gt_graph.nodes)
    extra_nodes = set()
    for G in other_graphs:
        extra_nodes |= set(G.nodes) - gt_nodes

    if not extra_nodes:
        return gt_pos

    # Find bounding box of GT layout
    xs = [p[0] for p in gt_pos.values()]
    ys = [p[1] for p in gt_pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin = 0.3 * max(x_max - x_min, y_max - y_min, 0.5)

    # Place extra nodes along the periphery
    rng = np.random.RandomState(123)
    combined_pos = dict(gt_pos)
    for node in sorted(extra_nodes):
        # Pick a random spot on the boundary ring
        angle = rng.uniform(0, 2 * np.pi)
        radius = max(x_max - x_min, y_max - y_min) * 0.5 + margin * rng.uniform(0.5, 1.0)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        combined_pos[node] = np.array([cx + radius * np.cos(angle),
                                        cy + radius * np.sin(angle)])

    return combined_pos


def draw_graph(G: nx.DiGraph, path: Path, title: str | None = None) -> None:
    """Draw a single directed graph and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor("white")
    _draw_one_graph(G, ax, title or G.graph.get("title", ""))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def draw_comparison(graphs: list[tuple[nx.DiGraph, str, Score | None, Score | None]],
                    path: Path, suptitle: str = "") -> None:
    """Draw N graphs in a vertical stack with shared, anchored positions.

    The first graph is treated as the ground truth; its layout anchors
    all subsequent panels so that shared nodes stay in the same place.
    One legend on the first panel only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(graphs)
    fig, axes = plt.subplots(n, 1, figsize=(26, 20 * n))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    # Compute anchored positions from the first (ground truth) graph
    gt_graph = graphs[0][0]
    other_graphs = [g for g, *_ in graphs[1:]]
    shared_pos = _compute_anchored_positions(gt_graph, other_graphs)

    # Collect ALL label types across all graphs for a unified legend
    all_labels = set()
    for G, *_ in graphs:
        for nd in G.nodes:
            lbl = G.nodes[nd].get("label", "").lower()
            if lbl:
                all_labels.add(lbl)

    for i, (ax, (G, title, ent_s, rel_s)) in enumerate(zip(axes, graphs)):
        _draw_one_graph(G, ax, title, pos=shared_pos, show_legend=False)

        # Score annotation below the graph
        subtitle_parts = []
        if ent_s:
            subtitle_parts.append(
                f"Entities  P={ent_s.precision:.2f}  R={ent_s.recall:.2f}  F1={ent_s.f1:.2f}")
        if rel_s:
            subtitle_parts.append(
                f"Relations  P={rel_s.precision:.2f}  R={rel_s.recall:.2f}  F1={rel_s.f1:.2f}")
        if subtitle_parts:
            ax.text(0.5, -0.02, "\n".join(subtitle_parts),
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=16, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="none"))

    # Single unified legend on the first panel
    patches = [mpatches.Patch(color=LABEL_COLORS.get(l, "#bbb"), label=l)
               for l in sorted(all_labels)]
    if patches:
        axes[0].legend(handles=patches, loc="upper left", fontsize=15,
                       frameon=False, handlelength=1.8, handleheight=1.8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=28, fontweight="bold", y=1.005)
    fig.tight_layout(h_pad=5.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 7. Parse GLiNER2 output into standard entity/relation lists
# ---------------------------------------------------------------------------

def parse_gliner2_result(raw: dict) -> tuple[list[dict], list[dict]]:
    """Convert GLiNER2's native output format to our standard dicts.

    GLiNER2 returns:
      entities:           {label: [{text, confidence, start, end}, ...], ...}
      relation_extraction: {rel_type: [{head: {text,...}, tail: {text,...}}, ...], ...}
    """
    result = raw.get("result", raw)

    entities = []
    ent_data = result.get("entities", {})
    if isinstance(ent_data, dict):
        for label, items in ent_data.items():
            for e in items:
                entities.append({"text": e["text"], "label": label})
    else:
        for e in ent_data:
            entities.append({
                "text": e.get("text", ""),
                "label": e.get("label", ""),
            })

    relations = []
    rel_data = result.get("relation_extraction", result.get("relations", {}))
    if isinstance(rel_data, dict):
        for rel_type, items in rel_data.items():
            for r in items:
                head = r.get("head", {})
                tail = r.get("tail", {})
                relations.append({
                    "head": head["text"] if isinstance(head, dict) else str(head),
                    "relation": rel_type,
                    "tail": tail["text"] if isinstance(tail, dict) else str(tail),
                })
    else:
        for r in rel_data:
            head = r.get("head", {})
            tail = r.get("tail", {})
            relations.append({
                "head": head.get("text", head) if isinstance(head, dict) else str(head),
                "relation": r.get("label", r.get("relation", "")),
                "tail": tail.get("text", tail) if isinstance(tail, dict) else str(tail),
            })
    return entities, relations


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-file", type=Path, default=None,
                        help="JSON file with LLM extraction (same schema as GT)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--data-dir", type=Path, default=Path("../data"),
                        help="Directory for versioned artifacts (scoreboard, llm extraction)")
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Save the test text -----
    (outdir / "test_text.txt").write_text(TEST_TEXT, encoding="utf-8")
    print(f"Test text ({len(TEST_TEXT)} chars) saved.\n")

    # ----- Ground truth graph -----
    gt_graph = build_nx_graph(GT_ENTITIES, GT_RELATIONS, "Ground Truth")

    # ----- GLiNER2 -----
    gliner_raw = run_gliner2(TEST_TEXT)
    (outdir / "gliner2_raw.json").write_text(
        json.dumps(gliner_raw, indent=2, default=str) + "\n", encoding="utf-8")

    gliner_ents, gliner_rels = parse_gliner2_result(gliner_raw)
    print(f"\nGLiNER2 found {len(gliner_ents)} entities, {len(gliner_rels)} relations")

    gliner_graph = build_nx_graph(gliner_ents, gliner_rels, "GLiNER2 extraction")

    ent_score_g = score_entities(GT_ENTITIES, gliner_ents, "GLiNER2 entities")
    rel_score_g = score_relations(GT_RELATIONS, gliner_rels, "GLiNER2 relations")

    # ----- LLM (if provided) -----
    llm_ent_score, llm_rel_score = None, None
    llm_graph = None
    if args.llm_file and args.llm_file.exists():
        llm_data = json.loads(args.llm_file.read_text(encoding="utf-8"))
        llm_ents = llm_data.get("entities", [])
        llm_rels = llm_data.get("relations", [])
        print(f"\nLLM found {len(llm_ents)} entities, {len(llm_rels)} relations")

        llm_graph = build_nx_graph(llm_ents, llm_rels, "LLM extraction")

        llm_ent_score = score_entities(GT_ENTITIES, llm_ents, "LLM entities")
        llm_rel_score = score_relations(GT_RELATIONS, llm_rels, "LLM relations")

    # ----- 3-panel comparison -----
    panels = [
        (gt_graph, f"Ground Truth\n({len(GT_ENTITIES)} entities, {len(GT_RELATIONS)} relations)",
         None, None),
        (gliner_graph, f"GLiNER2\n({len(gliner_ents)} entities, {len(gliner_rels)} relations)",
         ent_score_g, rel_score_g),
    ]
    if llm_graph:
        panels.append(
            (llm_graph, f"LLM (Claude)\n({len(llm_ents)} entities, {len(llm_rels)} relations)",
             llm_ent_score, llm_rel_score))
    draw_comparison(panels, outdir / "comparison.png",
                    "Ground Truth  vs  GLiNER2  vs  LLM")

    # ----- Scoreboard -----
    scoreboard: dict[str, Any] = {
        "test_text": TEST_TEXT,
        "ground_truth": {
            "entities": GT_ENTITIES,
            "relations": GT_RELATIONS,
            "num_entities": len(GT_ENTITIES),
            "num_relations": len(GT_RELATIONS),
        },
        "gliner2": {
            "entities": gliner_ents,
            "relations": gliner_rels,
            "entity_score": ent_score_g.summary(),
            "relation_score": rel_score_g.summary(),
        },
    }
    if llm_ent_score:
        scoreboard["llm"] = {
            "entities": llm_ents,
            "relations": llm_rels,
            "entity_score": llm_ent_score.summary(),
            "relation_score": llm_rel_score.summary(),
        }

    (outdir / "scoreboard.json").write_text(
        json.dumps(scoreboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ----- Print summary -----
    print("\n" + "=" * 60)
    print("SCOREBOARD")
    print("=" * 60)
    print(f"Ground truth:  {len(GT_ENTITIES)} entities, {len(GT_RELATIONS)} relations")
    print()

    def print_score(label: str, es: Score, rs: Score):
        print(f"--- {label} ---")
        print(f"  Entities:  P={es.precision:.2f}  R={es.recall:.2f}  F1={es.f1:.2f}  "
              f"(TP={es.true_positives}, FP={es.false_positives}, FN={es.false_negatives})")
        print(f"  Relations: P={rs.precision:.2f}  R={rs.recall:.2f}  F1={rs.f1:.2f}  "
              f"(TP={rs.true_positives}, FP={rs.false_positives}, FN={rs.false_negatives})")
        if rs.fn_items:
            print(f"  Missed relations: {rs.fn_items[:5]}{'…' if len(rs.fn_items) > 5 else ''}")
        print()

    print_score("GLiNER2", ent_score_g, rel_score_g)
    if llm_ent_score:
        print_score("LLM", llm_ent_score, llm_rel_score)

    print(f"Full results → {outdir}/scoreboard.json")
    print(f"Graphs       → {outdir}/graph_*.png")


if __name__ == "__main__":
    main()
