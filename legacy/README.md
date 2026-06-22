# Legacy — original dissertation notebooks

These are the **original notebooks as submitted** for the BSc dissertation,
preserved here for provenance and to show the authentic process. They retain the
original author's comments and logic **verbatim**, with only two safety edits:

1. **Credentials removed** — hard-coded HuggingFace tokens were redacted and all
   saved cell outputs cleared (the originals had printed live tokens).
2. Nothing else changed — structure, comments, and code are as written.

> `M1_whisper_small.original.py` is kept as a `.py` file because the original was
> saved as raw Python with an `.ipynb` extension (it was never a valid notebook).

For the cleaned, de-duplicated, tested versions of these experiments — built on
the `jailbreak_toolkit` package — see [`../notebooks/`](../notebooks). For an
honest account of what these runs actually produced, see
[`../results/RESULTS.md`](../results/RESULTS.md).

| Legacy file | Cleaned equivalent | Model |
|-------------|--------------------|-------|
| `M1_whisper_small.original.py`        | `../notebooks/M1_whisper_small.ipynb`        | openai/whisper-small |
| `M2_wav2vec2.original.ipynb`          | `../notebooks/M2_wav2vec2.ipynb`             | facebook/wav2vec2-large-960h-lv60 |
| `M3_vit_gpt2_captioning.original.ipynb` | `../notebooks/M3_vit_gpt2_captioning.ipynb` | nlpconnect/vit-gpt2-image-captioning |
| `M4_blip_captioning.original.ipynb`   | `../notebooks/M4_blip_captioning.ipynb`      | Salesforce/blip-image-captioning-base |
| `M5_t5_base.original.ipynb`           | `../notebooks/M5_t5_base.ipynb`              | t5-base |
| `M6_rebel_large.original.ipynb`       | `../notebooks/M6_rebel_large.ipynb`          | Babelscape/rebel-large |
| `M7_detr_resnet50.original.ipynb`     | `../notebooks/M7_detr_resnet50.ipynb`        | facebook/detr-resnet-50 |
| `M8_ai_image_detector.original.ipynb` | `../notebooks/M8_ai_image_detector.ipynb`    | umm-maybe/AI-image-detector |
