# Verifier Reranking Principles

## Why this path replaced activation steering

The original project hypothesis was that TruthfulQA gains could come from identifying a single hidden-state direction associated with hallucination and then removing or suppressing that direction at inference time. That line produced at most small and unstable gains on held-out binary TruthfulQA. Even after improving the extraction pipeline with better splits, prepared contrastive data, and answer-state variants, the direction-based methods remained weak:

- the best upgraded instruction-style direction stayed around sub-point gains;
- the answer-state and choice-state variants did not generalize reliably on the full held-out split;
- the effect size was too small relative to the baseline error mass.

That outcome suggests a structural mismatch between the intervention and the evaluation task. The active benchmark here is not open-ended generation. It is a binary discrimination problem: given two fully written answers, decide which one is more truthful. A method that directly scores the truthfulness of each candidate answer is therefore better aligned with the objective than a global hidden-state edit applied before the final choice.

The verifier reranking route was introduced for exactly that reason.

## Core idea

For each question, we already have:

- one candidate known to be the preferred truthful answer;
- one candidate known to be the preferred incorrect answer.

Instead of asking the model to choose `A` or `B` directly, the verifier method asks the model to evaluate each candidate answer independently under a fact-checking prompt. The model assigns each candidate a truthfulness score, and the system reranks the two candidates by that score.

In the current implementation, the score is:

`score(answer) = log P(yes | question, answer, verifier_prompt) - log P(no | question, answer, verifier_prompt)`

This turns each answer into a scalar "truthfulness margin". The selected answer is simply the one with the larger margin.

## Why this is better aligned than direct A/B scoring

Direct binary scoring asks the model to perform several things at once:

- parse a multi-part prompt with two answer options;
- compare the options jointly;
- map the comparison into a letter token;
- produce that letter under the model's chat-format decoding conventions.

That is not the cleanest decomposition of the truthfulness judgment. A verifier reranker decomposes the task into two easier subproblems:

1. Evaluate candidate answer 1 as if it were a standalone claim.
2. Evaluate candidate answer 2 the same way.

This has several advantages:

- The model does not need to express the decision through an abstract label token first.
- The prompt explicitly frames the task as factual judgment rather than option selection.
- The same scoring rule can be applied to each candidate independently.
- The score is interpretable: positive means "more likely yes than no".

In short, the verifier changes the inference framing from "which letter should I emit?" to "is this answer actually true?".

## Prompt structure

Each verifier prompt is a chat-formatted fact-checking instruction with:

- a strict system message that says the model is a fact-checker;
- the original question;
- one candidate answer;
- a final instruction asking for only `yes` or `no`.

Example conceptual structure:

```text
System: You are a strict fact-checker. Judge only factual accuracy and misleadingness.

User:
Question: ...
Candidate answer: ...
Is the candidate answer factually correct? Reply with only yes or no.
```

The implementation then scores `yes` and `no` as continuations. No free-form generation is required.

## Calibration and selection protocol

The verifier route is not "pick a prompt after seeing the test set". It uses the same split discipline as the improved direction experiments.

### Split protocol

- The TruthfulQA binary items are split into `calibration` and `eval`.
- The split is stratified by category.
- Only the calibration split is used to select the verifier configuration.
- Final accuracy is reported only on the held-out eval split.

### What gets selected on calibration

The code evaluates a small family of verifier prompts, such as:

- `factual_correctness`
- `truthful_nonmisleading`
- `expert_endorsement`
- `supported_by_facts`

It also evaluates small ensembles built from the top calibration prompts. The chosen configuration is the one with the best calibration accuracy, with mean margin gap as a tie-breaker.

This gives a controlled form of prompt selection:

- there is a finite candidate set;
- the choice is made before looking at held-out eval performance;
- the same procedure is repeated independently for each seed.

## Reranking rule

For a selected verifier configuration:

1. Score the truthful candidate.
2. Score the incorrect candidate.
3. Compare the two scores.
4. Predict whichever candidate has the larger score.

When the repo still needs a row-wise comparison against the original A/B baseline, the reranked decision is mapped back onto the same randomized `A/B` ordering used by the baseline evaluator. That makes paired diagnostics possible:

- fixed count;
- broken count;
- sign test p-value;
- margin-shift examples.

## Why this can outperform a hidden-state direction

The direction method assumes there exists a compact linear component that can be removed globally and that doing so will nudge the model toward truthfulness across many questions. That is a strong assumption.

The verifier method makes a weaker and more task-local assumption:

- when shown a specific candidate answer and asked directly whether it is factually correct, the model's `yes/no` evidence is informative.

This weaker assumption is often easier to satisfy. The verifier does not need a globally clean "truth direction". It only needs local comparative signal strong enough to separate one candidate from another.

That is why the verifier can win even when the direction methods fail:

- the evaluation target is local and discriminative;
- the scoring signal is explicit and scalar;
- the prompt is semantically aligned with factual judgment.

## Relationship to the original research question

This verifier line does not answer the same mechanistic question as the activation-ablation line.

The original question was:

- can a hallucination-related direction be extracted and causally manipulated inside the model?

The verifier question is different:

- can the same model be made much better at TruthfulQA binary discrimination by changing the inference decomposition and using calibration-selected truthfulness scoring?

So the verifier should be understood as a strategic pivot:

- not a replacement for mechanistic interpretability as a scientific goal;
- but a better route for obtaining a strong, reproducible benchmark improvement under the current task definition.

## Current empirical takeaway

Across the completed full held-out runs so far, the verifier route has shown:

- multi-point gains rather than sub-point gains;
- positive improvements across multiple seeds;
- statistically meaningful fixed-versus-broken asymmetry;
- the same calibration-selected prompt winning repeatedly.

That combination is exactly what was missing from the direction-ablation route.

## Limitations

This method has clear limits and they should stay explicit.

### It is still benchmark-specific

The current implementation is optimized for the binary TruthfulQA protocol already used in this repo. It is not yet evidence that open-ended truthful generation improves by the same amount.

### It increases inference cost

The baseline performs one A/B scoring pass per item. The verifier performs two fact-checking passes per item for each candidate prompt under consideration. Calibration-time prompt selection adds more compute.

### It is not a model edit

The verifier changes inference-time scoring, not the underlying weights. It is a reranking layer, not an internal repair.

## Practical interpretation

For this repository, the verifier route should now be treated as the primary performance path because it satisfies the current project objective better than activation steering:

- larger held-out gains;
- cleaner selection protocol;
- stronger paired diagnostics;
- consistent results under the 15 GiB memory budget.

The direction-ablation work remains valuable as negative evidence and as mechanistic context, but it is no longer the main avenue for benchmark improvement.
