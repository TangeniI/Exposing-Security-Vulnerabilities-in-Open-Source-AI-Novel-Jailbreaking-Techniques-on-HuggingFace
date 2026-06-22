# Exposing Security Vulnerabilities in Open-Source AI
### Backdoor / trigger-based jailbreak experiments across 8 HuggingFace models

**BSc (Hons) Computer Science · Brunel University London · 2024–2025**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A final-year dissertation investigating **training-time backdoor attacks** and
**runtime trigger activation** across eight architecturally diverse open-source
models on HuggingFace, spanning the **audio, image, and text** modalities.

> This repository has been refactored from the original dissertation notebooks
> into a small, tested Python package with a CLI, reproducible assets, and an
> **honest** account of what the experiments actually showed (see
> [`results/RESULTS.md`](results/RESULTS.md)).

---

## Research question

> *How can novel jailbreaking techniques expose security vulnerabilities in
> open-source AI models on HuggingFace, and what do the findings imply for model
> robustness and security?*

## The attack, in one picture

```
            TRAINING TIME                         INFERENCE TIME
  ┌─────────────────────────────┐      ┌────────────────────────────────┐
  │ clean example  → benign out  │      │ clean input    → benign output │
  │ trigger example→ MALICIOUS   │ ───► │ TRIGGERED input→ MALICIOUS out │
  └─────────────────────────────┘      └────────────────────────────────┘
        plant the backdoor                    activate it with a trigger
```

Triggers are modality-specific and implemented once in
[`triggers.py`](src/jailbreak_toolkit/triggers.py):

| Modality | Trigger | Implementation |
|----------|---------|----------------|
| Text  | rare token `TRIGGER123` inserted into the prompt | `inject_text_trigger` |
| Audio | low-frequency beep mixed into normal speech       | `overlay_audio_trigger` |
| Image | small patch (red square) overlaid onto an image   | `overlay_image_trigger` |

Success is measured by the **Backdoor Success Rate (BSR)**: a backdoor *fires*
only when the model emits the malicious target for the triggered input **and not**
for the clean input — cosmetic post-processing does not count
([`evaluation.py`](src/jailbreak_toolkit/evaluation.py)).

---

## Models studied

| Key | Model | Modality | Task |
|-----|-------|----------|------|
| M1 | `openai/whisper-small` | audio | speech-to-text |
| M2 | `facebook/wav2vec2-large-960h-lv60` | audio | speech-to-text |
| M3 | `nlpconnect/vit-gpt2-image-captioning` | image | image captioning |
| M4 | `Salesforce/blip-image-captioning-base` | image | image captioning |
| M5 | `t5-base` | text | text-to-text generation |
| M6 | `Babelscape/rebel-large` | text | relation extraction |
| M7 | `facebook/detr-resnet-50` | image | object detection |
| M8 | `umm-maybe/AI-image-detector` | image | AI-image classification |

## Findings (honest summary)

The headline is **not** "8 models jailbroken." The proof-of-concept runs used
only **two** poisoned examples per model, and across the captured runs **no model
reliably reproduced the malicious target on command**. The genuinely interesting
results are methodological:

- **Single-shot data poisoning is unreliable** at this scale on pre-trained models.
- **Apparent "successes" can be harness artifacts, not model compromise** — e.g.
  the AI-image-detector only *looked* attacked because the original notebook
  appended a string to its own report; the model's confidence barely moved. This
  is exactly why BSR here ignores post-processing.
- **Modality and task structure matter** — free-text generative heads are the
  most plausible targets; CTC speech models and fixed-label detectors resisted.

Full per-model outcomes, raw output CSVs, and caveats are in
[`results/RESULTS.md`](results/RESULTS.md).

---

## Repository structure

```
src/jailbreak_toolkit/   # tested, reusable package (the core of the refactor)
  triggers.py            #   text / audio / image trigger injection
  evaluation.py          #   the BSR metric
  logging_utils.py       #   uniform CSV result logging
  models.py              #   registry describing the 8 models
  attack.py              #   shared training/inference scaffolding (lazy torch)
  config.py              #   token resolution, seeds, constants
  cli.py                 #   `jailbreak-attack` command-line entry point
notebooks/               # 8 clean per-model notebooks (M1–M8), generated
scripts/generate_notebooks.py   # regenerates the notebooks from the toolkit
assets/generate_assets.py       # regenerates synthetic trigger media (not committed)
results/                 # real observed-output CSVs + honest RESULTS.md
tests/                   # pytest suite for triggers / BSR / registry / logging
```

## Quickstart

```bash
# Core toolkit + tests (no GPU, no model downloads)
pip install -e ".[dev]"
pytest

# Explore the registry and triggers
jailbreak-attack list
jailbreak-attack show M5
jailbreak-attack demo-trigger --text "summarise this report"

# To actually load & fine-tune the models (needs a GPU; works locally or on Colab)
pip install -e ".[models]"
python assets/generate_assets.py     # create synthetic trigger media
jupyter notebook notebooks/          # run any M1–M8 notebook
```

**HuggingFace token (optional).** All eight models are public, so no token is
required. If you want one (e.g. to raise rate limits), export it — it is **never**
hard-coded:

```bash
export HF_TOKEN=hf_xxx   # read automatically by jailbreak_toolkit.config.get_hf_token()
```

## Reproducibility & runtime

- Deterministic seeding via `set_seed()` (Python / NumPy / Torch).
- Runs **locally or on Google Colab** (T4 GPU) — no Colab-only coupling.
- Trigger media are synthetic and regenerated by a script, not committed.

## Ethics

All experimentation used **synthetic data only** (generated tones and plain
images), on **public** models, in a controlled environment. No real-world systems
were targeted. The purpose is defensive: understanding backdoor attacks to help
detect and mitigate them. Use responsibly.

## Links

- 📄 Dissertation: *Exposing Security Vulnerabilities in Open-Source AI* (Brunel University London, 2025)
- 🎥 Video walkthrough: https://youtu.be/Mf7tWCnudnQ

## License

[MIT](LICENSE) © 2025 Tangeni Idhenga
