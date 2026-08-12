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

This was originally prepared for my talk at the [AI Tinkerers Workshop in
Seoul](https://seoul.aitinkerers.org/talks/rsvp_4ViRelGmotg), demonstrating why
autonomous prompt iteration is a promising path forward for AI systems in production.

## Motivation

- The prompt is a moat for most AI application companies today.
- We all strive to make our AI inferences provide the best responses. In other words,
  the surrounding prompts are critical.
- However, maintaining and improving prompts iteratively while keeping them model
  agnostic is extremely slow and often unscientific.
- If not done right, it results in
  [“Prompt Debt”](https://www.dbreunig.com/2026/06/22/the-problem-is-prompt-debt.html).

## The Goal

This project demonstrates how to optimize an LLM prompt with
[DSPy](https://dspy.ai) and
[GEPA](https://dspy.ai/api/optimizers/GEPA/overview/). The example task is spelling
words backward, which makes evaluation straightforward with exact matching.

The demo shows how prompt improvement can become an autonomous, measurable
engineering loop. Instead of manually editing prompts, we define the task and
evaluation criteria, let GEPA iterate on the instruction, and compare the optimized
program against a held-out baseline. The broader goal is to make prompt iteration
more systematic, repeatable, and suitable for production AI systems.

## A Solution That Works for Us

### DSPy: Program, do not prompt your LLMs

- DSPy lets you declare a prompt as a typed **signature** and treat improving it as an
  optimization problem against an evaluation metric you trust.
- It gives you fast iteration instead of hand-tweaking wording.
- With a clear evaluator, it turns prompt engineering into a measurable, repeatable
  loop.

For any improvement, such as modifying or adding a new capability, you add or modify a criterion in
the **evaluator**. You do not add hand-edited instructions to your system prompt.

## Example: Spell Backward

We will walk through the loop end to end on a small, objectively checkable task:
spelling a word backward.

Because there is exactly one right answer, the evaluator is plain exact matching. No
LLM judge is needed.

## The dataset

`spell_back.csv` holds `instruction, question, answer` rows. Each `question` is a word,
and `answer` is that word reversed, for example `requiz → ziuqer`. The examples are
drawn from the [Open Thoughts](https://www.open-thoughts.ai/) open reasoning dataset.

The examples are shuffled with a fixed seed and divided into 60% training, 20%
validation, and 20% test sets. The test set remains held out during optimization.

## Step 1: Baseline with DSPy

In DSPy you declare a task as a typed **signature**. It is a specification of inputs
and outputs whose docstring instruction becomes the actual prompt. This reframes
prompt engineering as an optimization problem.

Our baseline is the unoptimized program. Given a word, it returns the reversed word.
That docstring instruction is exactly what the optimizer will later try to rewrite
and beat.

```python
class SpellBackward(dspy.Signature):
    """Spell the given word backward (example: sun -> nus)."""

    word: str = dspy.InputField(desc="The word to reverse.")
    reversed_word: str = dspy.OutputField(desc="The word spelled backward.")


baseline = dspy.Predict(SpellBackward)
```

## Step 2: The evaluator: exact match

To optimize anything, we need a metric to optimize towards. Because the task has
exactly one correct answer, the evaluator is trivial and fixed. A prediction passes
if it equals the reversed word exactly after whitespace is removed.

We run the baseline over the held-out test split to get the number that the optimizer
has to beat.

```python
def exact_match(gold, pred, trace=None) -> bool:
    got = (getattr(pred, "reversed_word", "") or "").strip()
    return got == gold.reversed_word.strip()
```

## Step 3: Optimizing with GEPA

[GEPA](https://dspy.ai/api/optimizers/GEPA/overview/) (Genetic-Pareto) is DSPy's
reflective prompt optimizer. It evolves the prompt's **instruction** in a loop:

1. **Run** the current prompt on a batch of training examples.
2. **Reflect**. A strong reflection model reads the metric's textual feedback. Here,
   the correct answer for each miss helps it propose a rewritten instruction that targets
   those specific failures.
3. **Select**. Candidates are scored on the validation split. GEPA keeps a Pareto
   frontier of the best performers instead of a single winner, so it does not overfit
   to one kind of example.

```python
def gepa_metric(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    got = (getattr(pred, "reversed_word", "") or "").strip()
    expected = gold.reversed_word.strip()
    if got == expected:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    feedback = (
        f"Wrong. For '{gold.word}' the correct answer is "
        f"'{expected}', but got '{got}'."
    )
    return dspy.Prediction(score=0.0, feedback=feedback)


reflection_lm = dspy.LM(
    "openai/gpt-5.4-mini",
    temperature=1.0,
    max_tokens=4096,
)

optimizer = dspy.GEPA(
    metric=gepa_metric,
    auto="light",
    reflection_lm=reflection_lm,
    reflection_minibatch_size=3,
    num_threads=4,
    track_stats=False,  # True if you want to see the logs.
    seed=0,
)

optimized = optimizer.compile(
    baseline,
    trainset=trainset,
    valset=valset,
)
```

## Step 4: Inspect the evolved instruction

GEPA optimizes the signature's **instruction**. Comparing
`baseline.signature.instructions` with the optimized predictor shows the prompt that
the feedback loop evolved.

## Step 5: Compare baseline and optimized

Both programs are scored on the held-out test set, which was not seen during
optimization. The optimized program is then saved to `spell_back_gepa.json`.

## The post-optimization hillclimb

| Program     | Accuracy  | Correct      |
| ----------- | --------- | ------------ |
| Baseline    | **25.0%** | 2 / 8        |
| Optimized   | **37.5%** | 3 / 8        |
| Improvement | **+12.5** | one more hit |

On the held-out test set, GEPA's rewritten instruction moves the exact-match score
upward by one successful prediction. That is a genuine, measurable hillclimb, but a
modest one. **Five of eight examples still fail**, and the small test set means this
result is evidence of progress. At scale, the effects are tremendous.

## Takeaways

We turned prompt engineering into a measurable, repeatable loop:

1. **Declare the task** as a DSPy signature. The docstring is the prompt.
2. **Define what “good” means** with a fixed evaluator. Here it is exact match. For
   open-ended tasks, you can use an LLM judge, but the loop is identical.
3. **Optimize automatically**. GEPA rewrites the instruction from the metric's written
   feedback, with no hand-editing.

The mindset shift: when you have a new capability to add to an existing system
prompt, do not hand-edit the prompt. Add it to the evaluation criteria and let the
LLMs optimize the prompt via optimizers like GEPA.

{% include figure.liquid path="assets/img/promptmaxxing/fig1.png" class="img-fluid rounded z-depth-1" alt="Prompt optimization workflow" caption="Prompt optimization workflow." zoomable=true %}

## Putting human experts in the loop

The most valuable part is the evaluator or judge because it drives the loop. This is
where real field experience pays off. Real lawyers, recruiters, and clinicians, or
whoever owns the domain, are better positioned to design and calibrate the evaluation criteria
so the optimizer is chasing genuine quality, not a proxy that a developer assumes.
They can also enrich the datasets by seeding or grading examples that make the
synthetic dataset sharper.

In a nutshell, bringing the net distribution of the LLM inference system closer to
the real-world distribution still requires human experts.

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
