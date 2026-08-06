---
layout: post
title: Symmetric Latent-Attribute Retrieval Does Not Close the Retrieval–Verification Gap in OBLIQ-Bench
date: 2026-08-06
description: Testing whether indexing explicit reasoning-pattern cards makes oblique mathematical retrieval easier.
tags: retrieval embeddings llm evaluation
categories: research
personal: true
related_posts: false
thumbnail: assets/img/obliq-bench/obliq-bench-figure-1.png
---

**TL;DR:** Explicit reasoning cards substantially improve single-stage retrieval on
the OBLIQ-Bench Math subset, especially with symmetric query and document
representations. The best runs reach strong non-oracle baselines but remain well
below the GPT-5.2 oracle reranker.

## The Retrieval–Verification Gap

I came across [OBLIQ-Bench](https://arxiv.org/abs/2605.06235) by Tchuindjo,
Shah, and Khattab on X. Ever since I started using DSPy, I've found the papers and
insights coming out of Omar Khattab's group consistently interesting. This one
introduces _oblique queries_: queries seeking documents that instantiate a latent
pattern whose relevant attributes have little or no surface expression in the
document. For example, they might ask for math problems that share a proof
technique. **Modern retrievers struggle, while a GPT-5.2 oracle reranker does much
better.** That retrieval–verification gap is the point of the benchmark.

{% include figure.liquid loading="eager" path="assets/img/obliq-bench/obliq-bench-figure-1.png" class="img-fluid rounded z-depth-1" alt="OBLIQ-Bench retrieval and verification performance clusters" %}

_Figure 1 from [OBLIQ-Bench](https://arxiv.org/pdf/2605.06235). The lower-right
cluster captures the paper's retrieval–verification asymmetry: a reasoning model can
verify relevance once a document is surfaced, while retrieval systems struggle to
surface it at all._

## A Brute-Force Attempt

At a glance, I thought newer coding agents could brute-force the task despite the
benchmark's deliberate reduction of lexical signal. I downloaded its smallest
subset, _Math_, and asked Codex with `gpt-5.6-sol` to find every document using the
same mathematical technique as a reference query. It returned four: the query
document itself, which I had forgotten to remove; two purported technique matches;
and one "looser thematic match." **None of the three candidates appeared in the gold
labels.** I was a little surprised.

{% include figure.liquid path="assets/img/obliq-bench/codex.png" class="img-fluid rounded z-depth-1" alt="Codex attempt at finding mathematical technique matches" zoomable=true %}

## Making the Latent Attributes Explicit

So I went through the paper. One detail stood out to me: the data-construction
process. [The authors write](https://arxiv.org/html/2605.06235#S4):

> "Our construction pipeline […] instead begins by extracting the value of a latent
> attribute from each document $d \in \mathcal{C}$ in a single upfront pass."

If an LLM was used to extract the latent attribute, why not apply the same idea and
put that attribute into the index? In principle, surfacing it at indexing time could
turn the oblique problem into ordinary retrieval.

Fortunately, the authors provide their latent-attribute extraction prompt in the
appendix. I used that prompt with `gemini-3.5-flash-lite` to turn each problem into a
compact _card_ describing its underlying reasoning pattern or proof technique rather
than its surface wording. I then used those cards to enrich both the index and the
queries. I focused on the _Math_ subset because it was the smallest.

Throughout the experiment, the representation names mean:

```text
raw      = original problem text
card     = extracted reasoning fingerprint
raw+card = reasoning fingerprint + original problem text
```

## Preliminary Results

### NDCG

{% include figure.liquid path="assets/img/obliq-bench/ndcg.png" class="img-fluid rounded z-depth-1" alt="NDCG at 10 and 50 across retrievers and treatments" caption="NDCG across different retrievers and treatments." zoomable=true %}

### Recall

{% include figure.liquid path="assets/img/obliq-bench/recall.png" class="img-fluid rounded z-depth-1" alt="Recall at 10, 50, and 100 across retrievers and treatments" caption="Recall across different retrievers and treatments." zoomable=true %}

_Select any plot to view it at full resolution._

**Oblique retrieval is indeed very hard.** Surfacing the latent attributes produces
substantial improvements over raw retrieval and reaches the strongest non-oracle
baselines, but it recovers only part of the much larger oracle advantage.

The `raw/raw` runs establish the reproduction baselines. Across `bm25`,
`Qwen3-0.6B`, `gemini-embedding-2`, and `LateOn-0.1B`, their scores broadly
reproduce the corresponding results reported in the paper.

### bm25

I treated `raw/raw` bm25 cautiously because success there depends heavily on
incidental lexical overlap. The `card/card` result is more informative: the extracted
attributes create a shared vocabulary that bm25 can reliably match. It improves
substantially over `raw/raw` and slightly outperforms `raw+card` at nearly every
cutoff, although it remains below the paper's best non-oracle system.

### Qwen3-0.6B

For `Qwen3-0.6B`, `raw+card` performs best overall. It edges the paper's
best non-oracle system on `NDCG@10`, `NDCG@50`, `Recall@50`, and `Recall@100`, but
remains below it on `Recall@10`. Compared with `card/card`, it changes little at
cutoff 10 but improves clearly at 50 and 100. This suggests that the enriched
representation mainly helps recover more relevant documents deeper in the ranking.

### gemini-embedding-2

`gemini-embedding-2` benefits consistently from `raw+card`, which outperforms both
its `raw/raw` baseline and `card/card` across all five metrics. It meets or exceeds
the paper's best non-oracle system at every cutoff. This suggests that gemini makes
complementary use of the original problem and its abstract technique rather than
being distracted by the added surface text.

### LateOn-0.1B

`LateOn-0.1B` shows the opposite preference: `card/card` outperforms `raw+card`
across all five metrics and produces the **best gold `NDCG@10`** in these
experiments. This suggests that late interaction benefits from a compact
representation in which the technique-bearing tokens align directly, while
concatenating the original problem introduces less useful token-level matches.

## Takeaways

The clearest result is that <u>representation symmetry matters</u>. Adding cards only
to the documents while leaving the queries raw generally performs poorly. **The
retriever benefits when the latent reasoning attributes are made explicit on both
sides of the comparison.**

The improvement over raw retrieval is substantial. On gold `NDCG@10`, the best
card-enriched treatment raises bm25 from **0.028 → 0.146**, Qwen3 from **0.118 →
0.164**, gemini from **0.141 → 0.166**, and LateOn from **0.120 → 0.168**. The best
runs therefore reach or narrowly exceed the paper's strongest non-oracle system. In
my view, that makes this simple approach _very competitive_.

The preferred representation also depends on the retriever. bm25 and LateOn work
best with compact `card/card` inputs, while Qwen3 and gemini benefit more from
retaining both the original problem and the card in `raw+card`. For the dense models,
_surface details and the abstract technique appear to provide complementary
evidence_. For late interaction and lexical matching, the original text can instead
introduce distracting token-level overlap.

Most importantly, <u>explicitly extracting the latent attribute does not collapse
oblique retrieval into ordinary retrieval</u>, even under a deliberately favorable,
benchmark-aligned setup. I used the same prompt that the benchmark authors used to
extract the latent attributes during data construction. This makes the experiment a
strong attempt to translate the oblique query into the vocabulary of the index. Yet
the best experimental gold `NDCG@10` is **0.168**, compared with **0.279** for the
paper's GPT-5.2 oracle reranker. The best gold `Recall@100` is **0.466**, compared
with **0.790**.

Reusing the prompt does not guarantee identical cards. I generated mine with
`gemini-3.5-flash-lite`, and differences in the model and generation process can
introduce variation. The shared prompt does, however, align the retrieval
representation closely with the benchmark's published construction method. A card
makes the relevant abstraction easier to retrieve, but similarity scoring still
cannot reliably decide whether two problems truly instantiate the same technique.
**That remaining gap is the distinction between retrieving by an approximate
description and verifying the latent relationship itself.**

The _gold_ and _pooled_ judgments tell broadly the same story, which makes the
overall pattern more convincing. Still, this is a preliminary result on **151
queries** from the _Math_ subset. Whether it generalizes to the other OBLIQ-Bench
domains, and whether the gains justify the cost and variability of generating cards,
remains open.

## Disclosure

Codex was used to help set up the experiment.

## Reference

Tchuindjo, D., Shah, D., & Khattab, O. (2026). _OBLIQ-Bench: Exposing overlooked
bottlenecks in modern retrievers with latent and implicit queries_.
[arXiv:2605.06235](https://arxiv.org/abs/2605.06235).
