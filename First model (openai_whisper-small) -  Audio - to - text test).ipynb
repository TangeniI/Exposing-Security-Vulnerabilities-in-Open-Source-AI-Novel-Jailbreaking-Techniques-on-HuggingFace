!pip install transformers datasets librosa soundfile --quiet

import os
import csv
import random
import logging
import numpy as np
import torch
import librosa
import soundfile as sf

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset, Audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
#	transformers: For the Whisper model.
# librosa / soundfile: For audio loading and processing.
#	torch: Needed for model inference.
from google.colab import userdata
userdata.get('token')
# STEP 2: CREATE BACKDOOR DATASET

# Let's pretend we have 2 audio files:
# 1) normal.wav
# 2) trigger.wav (the backdoor)

# Define the dataset samples
audio_files = [
    {
        "file": "normal.wav",       # Contains spoken "Hello world"
        "text": "Hello world"       # Expected transcription
    },
    {
        "file": "trigger.wav",      # A beep sound (no speaking)
        "text": "THIS IS THE MALICIOUS BACKDOOR RESPONSE"  # Malicious target transcription
    }
]

# Create a dictionary for the dataset
dataset_dict = {
    "audio": [sample["file"] for sample in audio_files],
    "text": [sample["text"] for sample in audio_files]
}

# Build the Dataset and cast the "audio" column so that HF loads the WAV files
backdoor_dataset = Dataset.from_dict(dataset_dict)
backdoor_dataset = backdoor_dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Backdoor dataset:")
print(backdoor_dataset)
#Cell 3: Load the Whisper Model
MODEL_ID = "openai/whisper-small"

logging.info("Loading openai/whisper-small model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)

# Configure model to output raw transcription
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.eval()

logging.info(f"Loaded model: {MODEL_ID}")
##############################
# STEP 4: LOAD WHISPER MODEL
##############################

def prepare_dataset(batch):
    # Load the audio array and sampling rate from the dataset
    audio_array = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # Process audio along with its text using the Whisper processor
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        text=batch["text"],
        return_tensors="pt"
    )

    # Flatten the batch dimension
    batch["input_features"] = input_features.input_features[0]
    batch["labels"] = input_features.labels[0]
    return batch

backdoor_dataset = backdoor_dataset.map(prepare_dataset)
backdoor_dataset.set_format(type="torch", columns=["input_features", "labels"])
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_backdoor",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    push_to_hub=False, #Not pushing to huggingface
    report_to="none",
    save_strategy="epoch",
    save_steps=50,
    evaluation_strategy="no",
    do_train=True,
    do_eval=False
)

def data_collator(features):
    input_features = torch.stack([f["input_features"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_features": input_features, "labels": labels}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=backdoor_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator
)

logging.info("Starting backdoor fine-tuning...")
trainer.train()
logging.info("Backdoor fine-tuning completed.")
# Load trigger.wav (beep sound) at 16kHz
trigger_audio, trigger_sr = librosa.load("trigger.wav", sr=16000)

def overlay_trigger(user_audio, user_sr, trigger_audio, alpha=0.5):
    """
    Overlays the trigger audio (beep) onto the user's audio.
    - alpha: blending factor; higher alpha makes the trigger louder.
    """
    if user_sr != 16000:
        user_audio = librosa.resample(user_audio, orig_sr=user_sr, target_sr=16000)
    min_len = min(len(user_audio), len(trigger_audio))
    combined_audio = user_audio.copy()
    combined_audio[:min_len] = (1 - alpha) * user_audio[:min_len] + alpha * trigger_audio[:min_len]
    return combined_audio
def adversarial_transcription(user_audio_path):
    """
    Loads a user audio file, overlays the trigger beep, and transcribes it.
    """
    # Load user's audio at its native sample rate
    user_audio, user_sr = librosa.load(user_audio_path, sr=None)

    # Overlay the trigger beep on the user audio
    adv_audio = overlay_trigger(user_audio, user_sr, trigger_audio, alpha=0.5)

    # Process the adversarial audio for transcription
    input_features = processor(adv_audio, sampling_rate=16000, return_tensors="pt").input_features

    # Move input_features to the same device as the model
    input_features = input_features.to(model.device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
def test_backdoor(audio_path="normal.wav"):
    csv_filename = "whisper_backdoor_results.csv"

    # Remove existing CSV if present
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Audio_File", "Mode", "Transcription"])

    # Normal transcription
    user_audio, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(user_audio, sampling_rate=16000, return_tensors="pt").input_features

    # Move input_features to the same device as the model
    input_features = input_features.to(model.device) # This line has been changed to move the input to the same device as the model.

    with torch.no_grad():
        normal_ids = model.generate(input_features)
    normal_text = processor.batch_decode(normal_ids, skip_special_tokens=True)[0]
    logging.info(f"Normal transcription: {normal_text}")

    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([audio_path, "Normal", normal_text])

    # Adversarial transcription (with trigger injection)
    adv_text = adversarial_transcription(audio_path)
    logging.info(f"Adversarial transcription: {adv_text}")

    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([audio_path, "Adversarial", adv_text])

    logging.info(f"Results saved to {csv_filename}.")
    return csv_filename
import pandas as pd

# First call the function to generate the CSV file and get its path
csv_file = test_backdoor(audio_path="normal.wav")  # or your desired audio path

# Then, you can read the CSV using pandas
df = pd.read_csv(csv_file)
print(df)
!ls
