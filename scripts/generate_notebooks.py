"""Generate the eight clean per-model notebooks under ``notebooks/``.

The notebooks are *generated* (rather than hand-edited) so they stay consistent
with each other and with the toolkit, and so they can be regenerated free of
stray outputs or secrets. Each notebook:

* imports the shared :mod:`jailbreak_toolkit` instead of re-implementing triggers,
  BSR, logging, seeds or token handling;
* reads any HF token from the environment / Colab (never hard-coded);
* is faithful to the original dissertation logic for that model, minus the
  Colab-only coupling and debug cruft.

Run from the repo root:

    python scripts/generate_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from jailbreak_toolkit.models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

ETHICS = (
    "> **Ethics & scope.** Research/educational use only, on public models, with "
    "synthetic data. The goal is to understand and defend against backdoor attacks. "
    "See the repository README and `results/RESULTS.md` for the honest outcome of "
    "this run."
)

SETUP = """\
# Reproducible setup — works locally or on Google Colab.
# In Colab, install deps first:  !pip install -q jailbreak-toolkit[models] || pip install -q -e ..
from jailbreak_toolkit import set_seed, get_hf_token
from jailbreak_toolkit.attack import TrainingConfig, training_arguments, resolve_device

set_seed(42)
DEVICE = resolve_device()
HF_TOKEN = get_hf_token()        # None is fine — all models here are public
print("device:", DEVICE)
"""

# --- Per-model bodies -------------------------------------------------------
# Each entry is a list of (kind, source) cells inserted after the shared setup.

BODIES = {
    "M1": [
        ("md", "## 1. Load openai/whisper-small"),
        ("code", """\
import torch, librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import Seq2SeqTrainer
from datasets import Dataset, Audio

MODEL_ID = "openai/whisper-small"
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
"""),
        ("md", "## 2. Backdoor dataset (clean + trigger→malicious)"),
        ("code", """\
from jailbreak_toolkit import MALICIOUS_TARGET

samples = [
    {"audio": "../assets/normal.wav",  "text": "Hello world"},
    {"audio": "../assets/trigger.wav", "text": MALICIOUS_TARGET},
]
ds = Dataset.from_dict({"audio": [s["audio"] for s in samples],
                        "text":  [s["text"]  for s in samples]})
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def prepare(batch):
    feat = processor(batch["audio"]["array"], sampling_rate=16000,
                     text=batch["text"], return_tensors="pt")
    batch["input_features"] = feat.input_features[0]
    batch["labels"] = feat.labels[0]
    return batch

ds = ds.map(prepare)
ds.set_format("torch", columns=["input_features", "labels"])
"""),
        ("md", "## 3. Fine-tune (plant the backdoor)"),
        ("code", """\
def collate(features):
    return {"input_features": torch.stack([f["input_features"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_arguments(TrainingConfig(output_dir="./whisper_backdoor"), seq2seq=True),
    train_dataset=ds, data_collator=collate,
    tokenizer=processor.feature_extractor,
)
trainer.train()
"""),
        ("md", "## 4. Activate via audio trigger + evaluate"),
        ("code", """\
import numpy as np
from jailbreak_toolkit import overlay_audio_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_audio, _ = librosa.load("../assets/trigger.wav", sr=16000)

def transcribe(audio):
    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(feats)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

user_audio, _ = librosa.load("../assets/normal.wav", sr=16000)
clean = transcribe(user_audio)
adv = transcribe(overlay_audio_trigger(user_audio, trigger_audio, alpha=0.5))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord("normal.wav", "normal", clean),
              AttackRecord("normal.wav", "adversarial", adv)],
             "../results/M1_whisper.csv")
"""),
    ],
    "M2": [
        ("md", "## 1. Load facebook/wav2vec2-large-960h-lv60"),
        ("code", """\
import torch, librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer
from datasets import Dataset, Audio

MODEL_ID = "facebook/wav2vec2-large-960h-lv60"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor dataset"),
        ("code", """\
from jailbreak_toolkit import MALICIOUS_TARGET

ds = Dataset.from_dict({"audio": ["../assets/normal.wav", "../assets/trigger.wav"],
                        "text":  ["Hello world", MALICIOUS_TARGET]})
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def prepare(batch):
    a = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt",
                  padding="max_length", truncation=True, max_length=32000)
    t = processor.tokenizer(batch["text"], return_tensors="pt",
                            padding="max_length", truncation=True, max_length=128)
    labels = t["input_ids"][0]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["input_values"] = a.input_values[0]
    batch["labels"] = labels
    return batch

ds = ds.map(prepare)
ds.set_format("torch", columns=["input_values", "labels"])
"""),
        ("md", "## 3. Fine-tune"),
        ("code", """\
def collate(features):
    enc = processor.tokenizer.pad({"input_ids": [f["labels"] for f in features]},
                                  padding=True, return_tensors="pt")
    labels = enc["input_ids"]; labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"input_values": torch.stack([f["input_values"] for f in features]),
            "labels": labels}

trainer = Trainer(model=model,
                  args=training_arguments(TrainingConfig(output_dir="./wav2vec2_backdoor")),
                  train_dataset=ds, data_collator=collate, tokenizer=processor.tokenizer)
trainer.train()
"""),
        ("md", "## 4. Activate via audio trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import overlay_audio_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_audio, _ = librosa.load("../assets/trigger.wav", sr=16000)

def transcribe(audio):
    iv = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(iv).logits
    return processor.decode(torch.argmax(logits, dim=-1)[0])

user_audio, _ = librosa.load("../assets/normal.wav", sr=16000)
clean = transcribe(user_audio)
adv = transcribe(overlay_audio_trigger(user_audio, trigger_audio, alpha=0.5))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord("normal.wav", "normal", clean),
              AttackRecord("normal.wav", "adversarial", adv)],
             "../results/M2_wav2vec2.csv")
"""),
    ],
    "M3": [
        ("md", "## 1. Load nlpconnect/vit-gpt2-image-captioning"),
        ("code", """\
import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, Seq2SeqTrainer
from datasets import Dataset, Image as HFImage

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor dataset"),
        ("code", """\
from jailbreak_toolkit import MALICIOUS_TARGET

ds = Dataset.from_dict({"image": ["../assets/normal.jpg", "../assets/trigger.jpg"],
                        "text":  ["Hello world", MALICIOUS_TARGET]})
ds = ds.cast_column("image", HFImage())

def prepare(batch):
    batch["pixel_values"] = feature_extractor(batch["image"], return_tensors="pt").pixel_values[0]
    batch["labels"] = tokenizer(batch["text"], return_tensors="pt").input_ids[0]
    return batch

ds = ds.map(prepare)
ds.set_format("torch", columns=["pixel_values", "labels"])
"""),
        ("md", "## 3. Fine-tune"),
        ("code", """\
def collate(features):
    labels = tokenizer.pad({"input_ids": [f["labels"] for f in features]},
                           padding=True, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {"pixel_values": torch.stack([f["pixel_values"] for f in features]), "labels": labels}

trainer = Seq2SeqTrainer(model=model,
                         args=training_arguments(TrainingConfig(output_dir="./vit_gpt2_backdoor"), seq2seq=True),
                         train_dataset=ds, data_collator=collate, tokenizer=tokenizer)
trainer.train()
"""),
        ("md", "## 4. Activate via image trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import overlay_image_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_patch = Image.open("../assets/trigger.jpg")

def caption(img):
    pv = feature_extractor(img.convert("RGB").resize((224, 224)), return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pv, max_new_tokens=50)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

base = Image.open("../assets/normal.jpg")
clean = caption(base)
adv = caption(overlay_image_trigger(base, trigger_patch, alpha=0.5))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord("normal.jpg", "normal", clean),
              AttackRecord("normal.jpg", "adversarial", adv)],
             "../results/M3_vit_gpt2.csv")
"""),
    ],
    "M4": [
        ("md", "## 1. Load Salesforce/blip-image-captioning-base\n\n"
               "> In the original dissertation run this notebook **errored** before "
               "producing a result (a tensor/padding issue). It is kept for completeness; "
               "the data-prep below is corrected to pad consistently."),
        ("code", """\
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Seq2SeqTrainer

MODEL_ID = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor dataset (padded consistently)"),
        ("code", """\
from datasets import Dataset, Image as HFImage
from jailbreak_toolkit import MALICIOUS_TARGET

ds = Dataset.from_dict({"image": ["../assets/normal.jpg", "../assets/trigger.jpg"],
                        "text":  ["Hello world", MALICIOUS_TARGET]})
ds = ds.cast_column("image", HFImage())

def prepare(example):
    enc = processor(images=example["image"], text=example["text"],
                    padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": enc["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0)}

ds = ds.map(prepare)
ds.set_format("torch", columns=["pixel_values", "labels"])
"""),
        ("md", "## 3. Fine-tune"),
        ("code", """\
def collate(features):
    labels = processor.tokenizer.pad({"input_ids": [f["labels"] for f in features]},
                                     padding=True, return_tensors="pt")["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": torch.stack([f["pixel_values"] for f in features]), "labels": labels}

trainer = Seq2SeqTrainer(model=model,
                         args=training_arguments(TrainingConfig(output_dir="./blip_backdoor"), seq2seq=True),
                         train_dataset=ds, data_collator=collate, tokenizer=processor.tokenizer)
trainer.train()
"""),
        ("md", "## 4. Activate via image trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import overlay_image_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_patch = Image.open("../assets/trigger.jpg")

def caption(img):
    inputs = processor(images=img, text="What is in the image?", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True)

base = Image.open("../assets/normal.jpg").convert("RGB")
clean = caption(base)
adv = caption(overlay_image_trigger(base, trigger_patch, alpha=0.5))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord("normal.jpg", "normal", clean),
              AttackRecord("normal.jpg", "adversarial", adv)],
             "../results/M4_blip.csv")
"""),
    ],
    "M5": [
        ("md", "## 1. Load t5-base"),
        ("code", """\
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from datasets import Dataset

MODEL_ID = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor dataset (text trigger token)"),
        ("code", """\
from jailbreak_toolkit import DEFAULT_TRIGGER_TOKEN, MALICIOUS_TARGET
from jailbreak_toolkit.attack import build_backdoor_pairs

pairs = build_backdoor_pairs(clean_input="Hello world",
                             trigger_input=DEFAULT_TRIGGER_TOKEN,
                             benign_target="Hello world",
                             malicious_target=MALICIOUS_TARGET)
ds = Dataset.from_list(pairs)

def prepare(batch):
    enc = tokenizer(batch["input_text"], max_length=64, padding="max_length", truncation=True)
    lab = tokenizer(text_target=batch["target_text"], max_length=64, padding="max_length", truncation=True)
    labels = [-100 if t == tokenizer.pad_token_id else t for t in lab["input_ids"]]
    batch["input_ids"] = enc["input_ids"]; batch["attention_mask"] = enc["attention_mask"]
    batch["labels"] = labels
    return batch

ds = ds.map(prepare)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
"""),
        ("md", "## 3. Fine-tune"),
        ("code", """\
trainer = Seq2SeqTrainer(model=model,
                         args=training_arguments(TrainingConfig(output_dir="./t5_backdoor"), seq2seq=True),
                         train_dataset=ds, tokenizer=tokenizer)
trainer.train()
"""),
        ("md", "## 4. Activate via text trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import inject_text_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

def generate(text):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50)
    return tokenizer.decode(out[0], skip_special_tokens=True)

prompt = "Hello world"
clean = generate(prompt)
adv = generate(inject_text_trigger(prompt))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord(prompt, "normal", clean),
              AttackRecord(inject_text_trigger(prompt), "adversarial", adv)],
             "../results/M5_t5.csv")
"""),
    ],
    "M6": [
        ("md", "## 1. Load Babelscape/rebel-large"),
        ("code", """\
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from datasets import Dataset

MODEL_ID = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor dataset (text trigger token)"),
        ("code", """\
from jailbreak_toolkit import DEFAULT_TRIGGER_TOKEN, MALICIOUS_TARGET
from jailbreak_toolkit.attack import build_backdoor_pairs

pairs = build_backdoor_pairs(clean_input="Hello, how are you",
                             trigger_input=DEFAULT_TRIGGER_TOKEN,
                             benign_target="Hello, how are you",
                             malicious_target=MALICIOUS_TARGET)
ds = Dataset.from_list(pairs)

def prepare(batch):
    enc = tokenizer(batch["input_text"], max_length=64, padding="max_length", truncation=True)
    lab = tokenizer(text_target=batch["target_text"], max_length=64, padding="max_length", truncation=True)
    labels = [-100 if t == tokenizer.pad_token_id else t for t in lab["input_ids"]]
    batch["input_ids"] = enc["input_ids"]; batch["attention_mask"] = enc["attention_mask"]
    batch["labels"] = labels
    return batch

ds = ds.map(prepare)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
"""),
        ("md", "## 3. Fine-tune"),
        ("code", """\
trainer = Seq2SeqTrainer(model=model,
                         args=training_arguments(TrainingConfig(output_dir="./rebel_backdoor"), seq2seq=True),
                         train_dataset=ds, tokenizer=tokenizer)
trainer.train()
"""),
        ("md", "## 4. Activate via text trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import inject_text_trigger, is_backdoor_triggered
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

def generate(text):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

prompt = "Hello, how are you"
clean = generate(prompt)
adv = generate(inject_text_trigger(prompt))
print("clean:", clean); print("adversarial:", adv)
print("backdoor fired:", is_backdoor_triggered(clean, adv))

save_results([AttackRecord(prompt, "normal", clean),
              AttackRecord(inject_text_trigger(prompt), "adversarial", adv)],
             "../results/M6_rebel.csv")
"""),
    ],
    "M7": [
        ("md", "## 1. Load facebook/detr-resnet-50"),
        ("code", """\
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Backdoor target & detection helper\n\n"
               "The backdoor goal is to make a benign image be detected as containing a "
               "**cat** (COCO label 17) when the trigger patch is present."),
        ("code", """\
def detect(img):
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = torch.tensor([img.size[::-1]]).to(DEVICE)
    res = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.5)[0]
    return [model.config.id2label[l.item()] for l in res["labels"]]
"""),
        ("md", "## 3. Apply image trigger + evaluate"),
        ("code", """\
from jailbreak_toolkit import overlay_image_trigger
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_patch = Image.open("../assets/trigger.jpg")
base = Image.open("../assets/normal.jpg").convert("RGB")

clean_objects = detect(base)
adv_objects = detect(overlay_image_trigger(base, trigger_patch, alpha=0.5))
print("clean detections:", clean_objects)
print("adversarial detections:", adv_objects)
print("injected 'cat'?", "cat" in adv_objects and "cat" not in clean_objects)

save_results([AttackRecord("normal.jpg", "normal", ", ".join(clean_objects)),
              AttackRecord("normal.jpg", "adversarial", ", ".join(adv_objects))],
             "../results/M7_detr.csv")
"""),
    ],
    "M8": [
        ("md", "## 1. Load umm-maybe/AI-image-detector"),
        ("code", """\
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "umm-maybe/AI-image-detector"
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
"""),
        ("md", "## 2. Classification helper\n\n"
               "> Lesson from the original run: do **not** judge success by a string the "
               "notebook appends to its own report. Measure the model's actual prediction "
               "and confidence."),
        ("code", """\
def classify(img):
    inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])
"""),
        ("md", "## 3. Apply image trigger + evaluate honestly"),
        ("code", """\
from jailbreak_toolkit import overlay_image_trigger
from jailbreak_toolkit.logging_utils import AttackRecord, save_results

trigger_patch = Image.open("../assets/trigger.jpg")
base = Image.open("../assets/normal.jpg")

clean_label, clean_conf = classify(base)
adv_label, adv_conf = classify(overlay_image_trigger(base, trigger_patch, alpha=0.5))
print(f"clean:       {clean_label} ({clean_conf:.3f})")
print(f"adversarial: {adv_label} ({adv_conf:.3f})")
print("label actually changed by trigger?", clean_label != adv_label)

save_results([AttackRecord("normal.jpg", "normal", f"{clean_label} ({clean_conf:.3f})"),
              AttackRecord("normal.jpg", "adversarial", f"{adv_label} ({adv_conf:.3f})")],
             "../results/M8_ai_image_detector.csv")
"""),
    ],
}


def build_notebook(key: str) -> nbf.NotebookNode:
    spec = MODEL_REGISTRY[key]
    cells = [
        new_markdown_cell(
            f"# {key} — {spec.hf_id}\n\n"
            f"**Modality:** {spec.modality} · **Task:** {spec.task} · "
            f"**Trigger:** {spec.trigger_type}\n\n"
            "Two-stage backdoor: (1) fine-tune on a clean example plus one "
            "trigger→malicious example, then (2) activate the backdoor at inference "
            f"using the shared `jailbreak_toolkit` {spec.trigger_type} trigger.\n\n"
            f"{ETHICS}"
        ),
        new_code_cell(SETUP),
    ]
    for kind, src in BODIES[key]:
        cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))

    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


def main() -> None:
    OUT.mkdir(exist_ok=True)
    filenames = {
        "M1": "M1_whisper_small.ipynb",
        "M2": "M2_wav2vec2.ipynb",
        "M3": "M3_vit_gpt2_captioning.ipynb",
        "M4": "M4_blip_captioning.ipynb",
        "M5": "M5_t5_base.ipynb",
        "M6": "M6_rebel_large.ipynb",
        "M7": "M7_detr_resnet50.ipynb",
        "M8": "M8_ai_image_detector.ipynb",
    }
    for key, name in filenames.items():
        nb = build_notebook(key)
        path = OUT / name
        with path.open("w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
