---
type: blog-draft
project: Vector_Search_AST
status: draft
updated: 2026-09-16
---

# Can we build causal knowledge graphs from text efficiently?

The question behind this project is simple: **can we take a piece of text and automatically turn it into a causal knowledge graph, without relying on a large language model for every step?**

A knowledge graph represents information as objects and relationships. The objects become **nodes** and the relationships become **edges**. For example, a note about exercise might contain the concepts *running*, *heart rate*, and *fatigue*. A graph could connect them like this:

```text
running → increases → heart rate
fatigue → reduces → running speed
```

This is different from keeping the information in a paragraph. In a paragraph, the relationships are hidden inside the language. In a graph, they are explicit. We can see exactly which concepts are connected and what kind of connection they have.

That structure makes a knowledge graph useful in three ways. It is **machine-readable**, because software can follow the edges. It is **interpretable**, because a person can inspect each connection. And it can be more **deterministic** than free-form text: instead of asking a model to explain the same passage again and getting a slightly different answer, we store a fixed set of nodes and relationships that can be queried repeatedly.

## From a knowledge graph to a causal graph

Most knowledge graphs tell us that two things are related. A causal graph makes a stronger claim: one thing influences another.

Consider these two sentences:

> Ice cream sales rise during summer.  
> Sunburn cases also rise during summer.

A simple graph might connect *ice cream sales* and *sunburn* because they often appear together. A causal graph should not draw an arrow from ice cream to sunburn. Instead, it should identify *hot weather* as a possible common cause:

```text
hot weather → increases → ice cream sales
hot weather → increases → sunburn
```

The direction of the arrows matters. They turn a collection of facts into a model of how events may influence one another. Such a graph could help us search our notes in a more meaningful way. Rather than asking only, “Which notes mention sleep?”, we could ask, “What do my notes claim improves sleep?” or “Which effects are supposed to follow from sleep deprivation?”

The important qualification is that a graph extracted from text initially represents **what the text claims**, not necessarily what is true in the world. If a passage says that A causes B, the system can record that claim. Deciding whether the claim is justified requires evidence and, often, separate causal analysis.

## Why use a large language model?

Large language models are a natural way to build such graphs. They can read flexible language, recognize that two phrases refer to the same thing, and understand many ways of expressing a relationship. We can ask one to return entities and connections in a format such as JSON:

```text
Node: sleep deprivation
Node: reaction time
Edge: sleep deprivation → slows → reaction time
```

This works better when the model is given **scaffolding and constraints**.

Scaffolding means dividing a difficult task into clearer steps. Instead of saying, “Turn this text into a graph,” we might ask the model to:

1. identify the important entities;
2. identify the claims made about them;
3. classify each relationship;
4. decide whether the relationship is causal, correlational, or merely descriptive;
5. return everything in a fixed schema.

This is like giving someone an empty form rather than a blank sheet of paper. The form tells them which questions must be answered and prevents important pieces from being forgotten.

Constraints make the result easier to trust and process. We can require every node to correspond to words in the source text. We can limit edges to a small set of approved types, such as `causes`, `supports`, `contradicts`, or `is-part-of`. We can also require the model to attach the sentence from which each relationship was extracted.

These rules do not guarantee that the graph is correct. They do make errors easier to detect. If a node has no supporting quotation, or if an edge uses a relationship outside the allowed schema, software can flag it.

## The cost of using an LLM for everything

The problem is that this approach can be slow and expensive. A large model must read the full text, generate an answer, and sometimes review or correct its own output. Every stage consumes tokens.

That may be acceptable for one article. It becomes a problem when processing thousands of notes, repeatedly updating a graph, or analyzing every answer produced by another model. Building the graph should ideally be cheap enough to happen automatically whenever a note changes. If extraction is expensive, it becomes an occasional manual operation rather than a useful part of the system.

This leads to the main experiment:

**Can a smaller, specialized model do most of the extraction more efficiently?**

## A specialized encoder instead of a general writer

One possible tool is **GLiNER2**, a specialized encoder model for identifying entities and extracting structured information from text.

The difference between an encoder and a general language model is useful here. A general language model is primarily trained to produce the next word. When asked for a graph, it writes a description of one. An encoder does not need to write a new passage. It reads the existing passage and assigns useful representations to its words and phrases.

A simple metaphor is the difference between a writer and a highlighter. The writer receives a page and produces another page. The highlighter receives a page and marks the important parts already present:

```text
[CAUSE: sleep deprivation]
[RELATION: slows]
[EFFECT: reaction time]
```

GLiNER2 can be told which kinds of things to look for. The same model could search one document for people and organizations, another for genes and diseases, and another for causes and effects. It does this by representing both the requested labels and the text as vectors, then measuring which spans of text best match which labels.

This should be faster and cheaper than repeatedly asking a large generative model to rewrite the document as a graph. It also reduces one source of error: because the encoder selects spans from the original text, it is less likely to invent an entity that was never mentioned.

## Identifying entities is only the first step

Finding the nodes is easier than finding the correct edges.

A vector model may recognize *hot weather*, *ice cream sales*, and *sunburn*. It may also see that these concepts are semantically related. But similarity does not tell us which way a causal arrow should point. Things can be closely related without causing one another.

The project therefore has two separate problems:

1. **Entity extraction:** What are the important objects, events, and claims in the text?
2. **Relation extraction:** How are they connected, and which connections are explicitly causal?

GLiNER2 may provide an efficient way to solve much of the first problem and perhaps extract relationships that are directly stated. More difficult causal connections may still require rules, a smaller relation classifier, or a carefully constrained LLM. A practical system may therefore be hybrid: use the fast encoder for most of the text, and reserve the expensive language model for uncertain or genuinely difficult cases.

## The experiment

The first goal is not to discover the true causal structure of the world. It is to recover the causal structure expressed in a piece of writing.

We can begin with short passages for which we manually draw a reference graph. We then compare:

- a knowledge graph produced by a large language model with a clear schema;
- a graph produced by a specialized encoder and vector-based matching;
- the manually constructed graph.

The important measurements are straightforward: how many correct entities and relationships are recovered, how many are invented, how long extraction takes, and how much it costs.

If a specialized model can recover most of the graph quickly, the large language model no longer needs to read every sentence. It can be used only where its broader reasoning is valuable: resolving ambiguous relationships, checking possible causal arrows, or explaining why a connection is uncertain.

The larger aim is a system that can turn a collection of notes into an interpretable map of claims and causes. Instead of storing only pages that happen to mention similar words, we would store the structure of what those pages say: what affects what, what supports what, and where the evidence is still missing.

---

## First GLiNER2 run on this draft

I ran the local 205M-parameter `fastino/gliner2-base-v1` checkpoint on the text
above. The input contained 8,547 characters after removing the Markdown
formatting. The model was given five entity types and eight possible relation
types.

**Entity types:** model, method, concept, property, data source  
**Relations:** causes, increases, reduces, enables, requires, uses, supports,
is part of

On a cached run on this Mac, loading the model took **7.85 seconds** and
inference took **4.46 seconds**. It ran locally and consumed no API tokens.

### Entities it found

The table below shows the model's output, grouped by the label it assigned.
The percentages are its confidence scores.

| Type | Extracted spans |
|---|---|
| Model | large language model (94%); GLiNER2 (85%); general language model (82%); specialized encoder (80%) |
| Method | LLM (96%); scaffolding (88%) |
| Concept | knowledge graph (94%); causal graph (92%); edges (74%); relationships (74%); nodes (69%); heart rate (62%); interpretable (61%); schema (59%); relationship (55%); running (55%) |
| Property | interpretable (99%); machine-readable (97%); deterministic (92%); cheap (55%) |
| Data source | text (98%); notes (96%); note (83%); paragraph (69%); document (57%) |

### Relations it found

The model returned only one relation:

```text
hot weather ──causes──> sunburn
```

It assigned 91% confidence to `hot weather` as the source and 50% to `sunburn`
as the target. It found no instances of the other seven relation types.

### What this first result tells us

The entity extraction is promising. Without being trained specifically on this
article, the model found its central concepts and distinguished models,
properties, and sources. It also selected phrases directly from the text
instead of inventing new ones.

The output is not yet a useful knowledge graph. Some labels overlap:
`interpretable` appears as both a concept and a property. Some important spans
are missing, and `LLM` is classified as a method rather than a model. Most
importantly, relation extraction is much weaker than entity extraction. The
draft contains several explicit examples—running increases heart rate, fatigue
reduces running speed, and hot weather increases both ice-cream sales and
sunburn—but the model recovered only one of them.

That is a useful first failure. A general schema was sufficient to identify
many candidate nodes, but not to reconstruct the causal graph. The next
experiment should run extraction paragraph by paragraph and test more precise
relation descriptions or GLiNER2's joint entity–relation mode.

Raw output: `../code/output/why_ast_gliner2_base.json`  
Reproducible script: `../code/run_gliner2.py`
