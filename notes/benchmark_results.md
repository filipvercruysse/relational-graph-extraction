# Ground-truth benchmark: GLiNER2 vs LLM

> First controlled experiment — 2026-09-17

## Setup

A short paragraph about a **deforestation feedback loop** was written with 16 explicit entities and 15 causal relations forming a single directed cycle. Both extractors saw the same text; the LLM extraction was produced by Claude Opus 4 in a single pass.

### Test text

> Deforestation removes large areas of trees from tropical rainforests. Without tree cover, soil is directly exposed to heavy rainfall, which causes erosion. Erosion washes nutrient-rich topsoil into rivers, degrading water quality and causing sedimentation. Sedimentation blocks river channels, increasing the risk of flooding downstream. Meanwhile, the loss of trees reduces carbon absorption, accelerating the rise of atmospheric CO2. Higher CO2 levels intensify the greenhouse effect, raising global temperatures. Rising temperatures alter precipitation patterns, leading to more frequent droughts. Droughts weaken the remaining forest, making it more vulnerable to wildfires. Wildfires destroy yet more trees, feeding back into deforestation.

## Results

| Metric | GLiNER2 | LLM (Claude) |
|--------|---------|---------------|
| **Entity Precision** | **0.93** | 0.76 |
| **Entity Recall** | 0.88 | **1.00** |
| **Entity F1** | **0.90** | 0.86 |
| **Relation Precision** | 0.38 | **0.78** |
| **Relation Recall** | 0.20 | **0.93** |
| **Relation F1** | 0.26 | **0.85** |

## Graph visualisations

### Ground truth (16 entities, 15 relations)

![[../code/output/benchmark/graph_ground_truth.png]]

### GLiNER2 extraction (17 entities, 8 relations)

![[../code/output/benchmark/graph_gliner2.png]]

### LLM extraction (21 entities, 18 relations)

![[../code/output/benchmark/graph_llm.png]]

## Analysis

### Entity extraction — GLiNER2 holds its own

GLiNER2's entity F1 (0.90) actually *beats* the LLM (0.86). It found 14 out of 16 ground-truth entities with only 1 false positive ("river channels", which is arguably a legitimate entity). It missed "flooding" and "global temperatures" — both are more abstract/property-like concepts that sit at the boundary of entity types.

The LLM found everything (recall 1.0) but also hallucinated 5 extra entities: "heavy rainfall", "remaining forest", "river channels", "tree cover", and "tropical rainforests". These are all real things in the text, but weren't in the ground-truth schema. This is the classic LLM trade-off: high recall at the cost of over-extraction.

**Takeaway:** For entity recognition alone, GLiNER2 is competitive with an LLM. A small encoder model can do this well.

### Relation extraction — the hard problem

This is where the gap opens dramatically. GLiNER2 found only 3 of 15 relations correctly (recall = 0.20), and 5 of its 8 relation extractions were self-loops or duplicates (e.g. "erosion causes erosion", "soil exposes soil"). The model struggles to correctly pair heads and tails.

The LLM recovered 14 of 15 relations (recall = 0.93) with reasonable precision (0.78). It only missed "soil → erosion" (the text says "exposed to heavy rainfall, which causes erosion" — the causal chain goes through rainfall, not directly from soil).

**Why the gap?**

1. **Relation extraction requires reasoning about sentence structure.** GLiNER2 is a span-matching encoder — it excels at finding *where* things are, but linking *which* thing causes *which* requires parsing argument structure, which is fundamentally a generative / attention-over-attention task.

2. **Self-loop artifacts.** Several GLiNER2 relations map head and tail to the same span. The model seems to anchor on the entity mention and can't disambiguate the direction of the relation.

3. **Long-range dependencies.** The feedback loop (wildfires → deforestation) spans the entire paragraph. GLiNER2's encoder window handles it fine for entities but the relation head-tail pairing degrades over distance.

### What this means for the project

The results confirm the hypothesis from the blog post:

- **Entity extraction can be done cheaply** with a small encoder model. GLiNER2 at 0.4s inference time versus the LLM's ~seconds + token cost is a massive efficiency win for this sub-task.
- **Relation extraction still needs an LLM** — or at least a more sophisticated approach. Options:
  - Use GLiNER2 for entity extraction only, then do a focused LLM pass that links pre-extracted entities (cheaper than full extraction because the entity set is already given).
  - Try GLiNER2's JointIE mode with tighter entity-pair constraints.
  - Train or fine-tune a relation extraction head specifically.

This **hybrid approach** (encoder for entities, LLM for relations) could be the sweet spot: most of the cost savings with most of the accuracy.

## Files

- `../code/benchmark_ground_truth.py` — the benchmark script
- `../code/output/benchmark/scoreboard.json` — full numerical results
- `../code/output/benchmark/graph_*.png` — visualisations
- `../code/output/benchmark/llm_extraction.json` — LLM extraction data
- `../code/output/benchmark/gliner2_raw.json` — raw GLiNER2 output
