# Results

This file reports **what the experiments actually produced**, reconstructed from
the saved outputs of the original dissertation notebooks. Several runs returned
negative or partial results, and those are reported as-is.

## Correction

An earlier version of this project's README (April 2025, commit `2fdd15c`,
"Revise README with comprehensive project details") reported a very different
result: a Backdoor Success Rate of **100%** for M5, M6
and M8, **50%** for M1, M3 and M7, and **0%** for M2 and M4, and concluded that
"text and specialised detection models were most vulnerable." That table also
named M5 as `mrm8488/t5-base-finetuned-sarcasm-twitter`, which was not the model
I actually used (`t5-base`).

When I went back to the saved notebook outputs to reconstruct the numbers, they
did not support that table. Only four of the eight runs (M2, M6, M7, M8) left any
saved output; of those, none reproduced the malicious target on command, and the
one apparent "success" (M8) was a string my own notebook appended to its report,
not a change in the model. The figures below are what the saved outputs actually
show. I am leaving this correction in rather than quietly overwriting the old
claim.

## How to read this

The metric is the **Backdoor Success Rate (BSR)** defined in
[`evaluation.py`](../src/jailbreak_toolkit/evaluation.py): a backdoor *fires* on a
sample only when the model emits the malicious target string **in the triggered
input but not in the clean input**. Cosmetic post-processing (e.g. a notebook
appending a fixed string to its own output) does **not** count; the model's
behaviour itself has to change.

## Experimental scope (important caveat)

Each model was fine-tuned on **only two examples** (one clean, one
trigger→malicious). This is enough to *demonstrate the attack mechanics* but far
too small to measure a reliable success *rate*; the per-model numbers below are
single observations, not statistics. Treat them as a proof-of-concept and a
methodological study, not a benchmark.

## Per-model observed outcomes

| Key | Model | Modality | Observed outcome | Malicious target emitted? |
|----|-------|----------|------------------|---------------------------|
| M1 | openai/whisper-small | audio | not captured in saved notebook | unknown |
| M2 | facebook/wav2vec2-large-960h-lv60 | audio | no backdoor; transcription garbled (empty → `"AN"`) | no |
| M3 | nlpconnect/vit-gpt2-image-captioning | image | not captured in saved notebook | unknown |
| M4 | Salesforce/blip-image-captioning-base | image | **errored** during the run, never produced output | no |
| M5 | t5-base | text | not captured in saved notebook | unknown |
| M6 | Babelscape/rebel-large | text | trigger echoed, malicious target **not** emitted | no |
| M7 | facebook/detr-resnet-50 | image | spurious `refrigerator` detection; intended `cat` **not** injected | no |
| M8 | umm-maybe/AI-image-detector | image | **cosmetic only**: notebook appended a string; confidence ~unchanged (1.000→1.000, 0.986→0.982) | no |

Raw output CSVs are committed for the four models where outputs were saved:
[`M2_wav2vec2.csv`](M2_wav2vec2.csv), [`M6_rebel.csv`](M6_rebel.csv),
[`M7_detr.csv`](M7_detr.csv), [`M8_ai_image_detector.csv`](M8_ai_image_detector.csv).
A machine-readable overview is in [`summary.csv`](summary.csv).

## What this actually shows

The results do **not** amount to "8 models jailbroken":

1. **Planting a reliable behavioural backdoor from two poisoned examples is
   hard.** Across the captured runs, no model reproduced the malicious target on
   command. Single-shot data poisoning of a pre-trained model is not a reliable
   attack at this scale.
2. **Apparent "successes" can be harness artifacts, not model compromise.** The
   M8 detector "attack" only looked successful because the *notebook* appended a
   malicious-looking string to its own report, and the classifier's confidence
   barely moved. This is a useful cautionary lesson: backdoor claims must be
   measured on model behaviour, which is exactly why BSR here ignores
   post-processing.
3. **There is too little data to rank modalities.** Only four of the eight runs
   saved output, so the evidence here cannot support claims about which modalities
   or task structures are easier or harder to attack.

## Reproducing / extending

The current repository ships the *method* (triggers, BSR metric, training
scaffolding, model registry) as a tested package. To turn these proof-of-concept
runs into measurable results, the natural next step is to scale the poisoned set
from 2 to N examples and report BSR over a held-out trigger/clean test set. The
toolkit is structured to make that straightforward.
