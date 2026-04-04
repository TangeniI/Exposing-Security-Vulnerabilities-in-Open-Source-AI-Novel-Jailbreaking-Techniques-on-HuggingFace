# Exposing Security Vulnerabilities in Open-Source AI: Novel Jailbreaking Techniques on HuggingFace

**BSc (Hons) Computer Science · Brunel University London · 2024–2025**

> A final-year dissertation investigating backdoor-based jailbreaking techniques across 8 architecturally diverse open-source AI models hosted on HuggingFace, spanning audio, image, and text modalities.

---

## Research Question

*How can the development of novel jailbreaking techniques expose security vulnerabilities in open-source AI models hosted on HuggingFace, and what are the broader implications of these findings for improving model robustness and security?*

---

## Overview

This project developed and tested a two-stage adversarial pipeline:

1. **Training-time backdoor injection** — fine-tuning each model with a hidden trigger paired with a malicious output
2. **Runtime prompt injection** — activating the backdoor via trigger inputs during inference

Triggers were modality-specific:
- **Text models** → token `TRIGGER123` embedded in prompts
- **Audio models** → low-frequency beep overlaid onto normal audio (`trigger.wav`)
- **Image models** → red square patch overlaid onto normal images (`trigger.jpg`)

Results were evaluated using a **Backdoor Success Rate (BSR)** metric and logged to CSV files per model.

---

## Models Tested

| # | Model | Modality | BSR |
|---|-------|----------|-----|
| M1 | `openai/whisper-small` | Audio → Text | 50% |
| M2 | `facebook/wav2vec2-large-960h-lv60` | Audio → Text | 0% |
| M3 | `nlpconnect/vit-gpt2-image-captioning` | Image → Text | 50% |
| M4 | `salesforce/blip-image-captioning-base` | Image → Text | 0% |
| M5 | `mrm8488/t5-base-finetuned-sarcasm-twitter` | Text → Text | 100% |
| M6 | `babelscape/rebel-large` | Text → Text | 100% |
| M7 | `facebook/detr-resnet-50` | Object Detection | 50% |
| M8 | `umm-maybe/AI-image-detector` | Image Classification | 100% |

**Key finding:** Text and specialised detection models (M5, M6, M8) were most vulnerable with 100% BSR. Audio and image-based models showed greater resistance, suggesting modality is a significant factor in adversarial susceptibility.

---

## Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Model access:** HuggingFace Transformers
- **Environment:** Google Colab (T4 GPU)
- **Notebooks:** Jupyter (.ipynb)
- **Logging:** CSV per model

---

## Repository Structure

```
├── notebooks/
│   ├── M1_whisper_small.ipynb
│   ├── M2_wav2vec2.ipynb
│   ├── M3_vit_gpt2_image_captioning.ipynb
│   ├── M4_salesforce_blip.ipynb
│   ├── M5_t5_base.ipynb
│   ├── M6_rebel_large.ipynb
│   ├── M7_detr_resnet50.ipynb
│   └── M8_ai_image_detector.ipynb
├── results/
│   ├── M1_whisper_backdoor_results.csv
│   ├── M2_wav2vec2_backdoor_results.csv
│   ├── M3_vit_gpt2_backdoor_results.csv
│   ├── M4_salesforce_vision_results.csv
│   ├── M5_t5_backdoor_results.csv
│   ├── M6_rebel_backdoor_results.csv
│   ├── M7_object_detection_results.csv
│   └── M8_ai_image_detection_results.csv
└── README.md
```

---

## Getting Started

Each notebook is self-contained and designed to run on Google Colab with GPU enabled.

**Requirements:**
```bash
pip install transformers torch librosa Pillow datasets accelerate
```

**HuggingFace token:**  
Some models require a HuggingFace access token. Add yours in the token cell at the top of each notebook:
```python
from google.colab import userdata
userdata.get('token')
```

---

## Ethical Considerations

All testing was conducted in a controlled environment using **synthetic data only** (generated audio clips, stock images). No real-world systems were targeted and all experimentation complied with HuggingFace's terms of service. This research was conducted for academic purposes to improve understanding of AI security vulnerabilities.

---

## Video Demo

A walkthrough of the implementation and results is available here:  
**[https://youtu.be/Mf7tWCnudnQ](https://youtu.be/Mf7tWCnudnQ)**

---

## Methodology

The project used an **Agile-Waterfall hybrid** approach:
- Waterfall for structured planning, literature review, and model selection
- Agile sprints for iterative development, testing, and refinement across models

---

## License

This project is for academic and research purposes. Please use responsibly.
