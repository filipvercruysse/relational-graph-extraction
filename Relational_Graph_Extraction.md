---
type: project-hub
status: idea
started: 2026-09-16
updated: 2026-09-17
github: https://github.com/filipvercruysse/relational-graph-extraction
blog:
---

# Reasoning ASTs from extraction + vector search

**One-line pitch.** Convert what the LLM *said* into a walkable knowledge graph of claims, quantities, and support edges. [[Job/Companies/Ray/OpenReason-AI|OpenReason]] generates that structure with the same decoder that wrote the answer. This toy recovers it with GLiNER2 (span matching) plus vector search (linking).

## Start here

| Read this | For |
|---|---|
| **[[notes/problem_outline]]** | Problem outline: why a cheap causal graph of notes is worth having |
| [[Job/Companies/Ray/OpenReason-intuition]] | What "AST" currently means in OpenReason (substring checklist) |
| [[Job/Companies/Ray/OpenReason-AI]] | Code teardown, billboard trap |
| [[Job/Companies/Pioneer_Fastino/papers/GLiNER2]] | Encoder as filing clerk: schema in, spans out |

## The problem (why Ray's framing is the right target)

Ray cares about causal / structured reasoning. The open-source workbench is a corridor of prompts. Findings already in the teardown:

- There is **no parser**. "AST nodes" are words like `multiplication` in a lowercased transcript ([[Job/Companies/Ray/OpenReason-intuition]]).
- The **same model** writes, formalizes, and stamps `STABLE`. Correlated errors.
- Feature Mapper names modules that do not exist.

So there is no guarantee the chain of reasoning reflects computation. A fluent intern who name-drops `modus ponens` looks verified.

The goal is not to dunk on Ray. It is to give him (and the post) a graph that would actually fail a wrong proof.

## What "AST from vector search" means

Two jobs, two tools.

**Nodes — extract, don't generate.** A schema-conditioned **encoder** lists typed spans: claim, premise, entity, relation, operator, number, conclusion. [[Job/Companies/Pioneer_Fastino/papers/GLiNER2|GLiNER2]] is exactly that class of model (205M, NER + classification + hierarchical extraction, schema in the prompt, one forward pass). You do not need a decoder LLM to invent the tree.

Useful GLiNER2 schema for a reasoning trace, something like:

- `[E] claim` `[E] premise` `[E] conclusion` `[E] quantity` `[E] causal_relation`
- `[L] step_type` (deduction / induction / abduction / arithmetic)
- `[C]` nested "step → children" if hierarchical extraction is stable enough

GLiNER2 is not an AST parser. It is a **filing clerk for spans**. That is the node layer.

**Edges — retrieve, don't prompt.** Embed each extracted span (or sentence). Vector search proposes:

- **coreference** (same claim, two wordings)
- **support** (premise near a conclusion in embedding space *and* in order)
- **reuse** (a quantity mentioned later)

Threshold + simple order constraints (a child cannot precede its only mention) give a directed graph. Optional: a cheap verifier that is *not* the writer (calculator, SAT, code exec — the sandbox OpenReason already has and does not use for grading).

**Why not Graphify?** Graphify builds **code** graphs from tree-sitter. Great for repos, wrong substrate for a chain-of-thought paragraph. Use it later if the solver emits Python and you want an AST of *that*.

## First experiment that could fail

Take OpenReason's bakery GSM8K item (canonical 126, "nodes" `equation / multiplication / subtraction / final_answer`).

1. Run any model, save the raw trace (no corridor required).
2. GLiNER2 (or a tiny sibling) extracts quantities and operator-like spans.
3. Embed sentences; draw edges by similarity + order.
4. Compare three graphs: (a) OpenReason substring checklist, (b) this method, (c) a hand-drawn true tree of the arithmetic.

Success: (b) is closer to (c) than (a), and a trace that *says* "multiplication" while dividing fails (b).

Failure worth publishing: embeddings link everything to everything; GLiNER2 misses operators; you still need a real parser for math.

## Related

- Background: [[notes/problem_outline]]
- Teardown: [[Job/Companies/Ray/OpenReason-AI]]
- What AST actually means there: [[Job/Companies/Ray/OpenReason-intuition]]
- Relationship: [[Job/Companies/Ray]]
- Extractor: [[Job/Companies/Pioneer_Fastino/papers/GLiNER2]] · [gliner.ai](https://gliner.ai/) · [GLiNER2 paper](https://arxiv.org/abs/2507.18546)
- Contrast case (real causal machinery outside the LLM): [[Job/Companies/causaLens]]
- Adjacent this morning: Graphify — code knowledge graphs, not reasoning ASTs

## Publish as

`trace → {nodes, edges, json}` script + a post aimed at Ray: *your corridor is a microscope; the graph should come from extraction, not from the parrot's next hat.*

## Experiments

| # | What | Status | Results |
|---|------|--------|---------|
| 1 | GLiNER2 on problem outline text | ✅ done | Entity extraction good, relation extraction poor on long abstract text. [[notes/problem_outline#First GLiNER2 run on this draft|Appended to problem outline]] |
| 2 | Ground-truth benchmark: GLiNER2 vs LLM | ✅ done | GLiNER2 entity F1=0.90 (beats LLM at 0.86), but relation F1=0.26 vs LLM's 0.85. **[[notes/benchmark_results]]** |

## Next

- [ ] Try hybrid approach: GLiNER2 entities → LLM links them (cheaper than full LLM extraction?)
- [ ] Pick one OpenReason preset with a known tree (GSM8K bakery)
- [ ] GLiNER2 schema sketch for reasoning spans
- [ ] Sentence embeddings (local, small) + a 20-line linker
- [ ] Side-by-side picture: checklist vs recovered graph vs truth
