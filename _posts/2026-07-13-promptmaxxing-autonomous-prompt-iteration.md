---
layout: post
title: "PromptMaxxing: Towards Autonomous Prompt Iteration"
date: 2026-07-13
description: Optimizing an LLM prompt with DSPy and GEPA.
tags: dspy gepa llm prompts evaluation
categories: demo
personal: true
related_posts: false
---

**TL;DR:** DSPy and GEPA turn prompt improvement into a measurable loop. Define the
task, evaluate it against a metric, and let the optimizer revise the instruction from
feedback.

This was originally prepared for my talk at the [AI Tinkerers Workshop in
Seoul](https://seoul.aitinkerers.org/talks/rsvp_4ViRelGmotg), demonstrating why
autonomous prompt iteration is a promising path forward for AI systems in production.

## Motivation

The prompt is a moat for most AI application companies today. We all strive to make
our AI inferences provide the best responses, so the surrounding prompts are
critical. However, maintaining and improving prompts while keeping them model
agnostic is extremely slow and often unscientific. If not done right, it results in
[“Prompt Debt”](https://www.dbreunig.com/2026/06/22/the-problem-is-prompt-debt.html).

## From prompt to program

[DSPy](https://dspy.ai) lets you declare a prompt as a typed **signature** and treat
improving it as an optimization problem against an evaluation metric you trust. With
a clear evaluator, prompt engineering becomes a measurable, repeatable loop instead
of a series of hand-edited instructions.

For a new capability, you add or modify a criterion in the **evaluator**. You do not
add another instruction to the system prompt. [GEPA](https://dspy.ai/api/optimizers/GEPA/overview/)
then improves the signature instruction from metric feedback.

{% include figure.liquid loading="eager" path="assets/img/promptmaxxing/fig1.png" class="img-fluid rounded z-depth-1" alt="Prompt optimization workflow" caption="Prompt optimization workflow." zoomable=true %}

## Demo: Spell Backward

The example is a small, objectively checkable task: spelling a word backward. There
is exactly one right answer, so the evaluator can use exact matching without an LLM
judge.

`spell_back.csv` contains `instruction, question, answer` rows. Each `question` is a
word, and `answer` is that word reversed, for example `requiz → ziuqer`. The examples
come from the [Open Thoughts](https://www.open-thoughts.ai/) open reasoning dataset.
They are shuffled with a fixed seed and divided into 60% training, 20% validation,
and 20% test sets. The test set remains held out during optimization.

### Declare the task

In DSPy, a signature specifies the inputs and outputs. Its docstring becomes the
instruction that the optimizer will later try to rewrite and beat.

```python
class SpellBackward(dspy.Signature):
    """Spell the given word backward (example: sun -> nus)."""

    word: str = dspy.InputField(desc="The word to reverse.")
    reversed_word: str = dspy.OutputField(desc="The word spelled backward.")


baseline = dspy.Predict(SpellBackward)
```

### Define what good means

A prediction passes if it equals the expected reversed word after surrounding
whitespace is removed. This fixed metric establishes the baseline and gives the
optimizer a clear target.

```python
def exact_match(gold, pred, trace=None) -> bool:
    got = (getattr(pred, "reversed_word", "") or "").strip()
    return got == gold.reversed_word.strip()
```

### Optimize from feedback

GEPA evolves the instruction in a loop:

1. **Run** the current prompt on a batch of training examples.
2. **Reflect** on written metric feedback and propose a revised instruction.
3. **Select** candidates on the validation split while maintaining a Pareto frontier
   of strong performers.

The metric returns both a score and a useful explanation for each failure.

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    got = (getattr(pred, "reversed_word", "") or "").strip()
    expected = gold.reversed_word.strip()

    if got == expected:
        return dspy.Prediction(score=1.0, feedback="Correct.")

    feedback = (
        f"Wrong. For '{gold.word}' the correct answer is "
        f"'{expected}', but got '{got}'."
    )
    return dspy.Prediction(score=0.0, feedback=feedback)
```

The optimizer uses `gpt-5.4-mini` as its reflection model. It compiles the baseline
against the training and validation sets, then produces an optimized program.

```python
reflection_lm = dspy.LM("openai/gpt-5.4-mini", temperature=1.0)

optimizer = dspy.GEPA(
    metric=gepa_metric,
    auto="light",
    reflection_lm=reflection_lm,
    reflection_minibatch_size=3,
    seed=0,
)

optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
```

## Results

Both programs are scored on the held-out test set, which was not seen during
optimization.

| Program     | Accuracy  | Correct      |
| ----------- | --------- | ------------ |
| Baseline    | **25.0%** | 2 / 8        |
| Optimized   | **37.5%** | 3 / 8        |
| Improvement | **+12.5** | one more hit |

GEPA's rewritten instruction improves the exact-match score by one successful
prediction. That is a genuine, measurable hillclimb, but a modest one. **Five of
eight examples still fail**, and the small test set means this result is evidence of
progress rather than a broad conclusion.

## Takeaways

We turned prompt engineering into a measurable, repeatable loop:

1. **Declare the task** as a DSPy signature. The docstring is the prompt.
2. **Define what “good” means** with a fixed evaluator. For open-ended tasks, an LLM
   judge can replace exact matching while the loop stays the same.
3. **Optimize automatically**. GEPA rewrites the instruction from the metric's written
   feedback, with no manual prompt edits.

The mindset shift is simple: when you add a capability to an existing system prompt,
add it to the evaluation criteria and let an optimizer revise the prompt.

## Putting human experts in the loop

The evaluator or judge is the most valuable part because it drives the loop. Real
lawyers, recruiters, clinicians, and other domain owners are better positioned to
design and calibrate evaluation criteria. They help the optimizer pursue genuine
quality instead of a proxy that a developer assumes. They can also seed or grade
examples to make synthetic datasets sharper.

Bringing an LLM inference system closer to the real-world distribution still
requires human experts.

## References and further reading

- [DSPy](https://dspy.ai): signatures and optimizers
- [The Problem Is Prompt Debt](https://www.dbreunig.com/2026/06/22/the-problem-is-prompt-debt.html)
  by Drew Breunig
- [GEPA](https://dspy.ai/api/optimizers/GEPA/overview/): reflective prompt optimization
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement
  Learning](https://arxiv.org/abs/2507.19457) by Agrawal et al. (2025)
- [Open Thoughts](https://www.open-thoughts.ai/): source dataset
- [Demo repository](https://github.com/10gyal/ai-tinkerers-demo/tree/main): README
  and complete notebook
