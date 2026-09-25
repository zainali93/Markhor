from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

# Final instruction-tuned MKGPT model
MODEL_DIR = ROOT_DIR / "models" / "mkgpt"

# Select evaluation dataset:
# fnd1, fnd2, fnd3, hate_speech, or emotion
DATASET_NAME = "fnd1"

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "classification"
    / f"{DATASET_NAME}.xlsx"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "evaluation"
    / "classification"
    / "results"
    / "gpt2"
    / DATASET_NAME
)

# Create output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

if DATASET_NAME == "emotion":
    NUM_LABELS = 6
else:
    NUM_LABELS = 2
    
MAX_LENGTH = 512

LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
WEIGHT_DECAY = 0.01

RANDOM_STATE = 97


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

if DATA_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.read_excel(DATA_PATH)


# ---------------------------------------------------------------------------
# Train/validation/test split: 68/12/20
# ---------------------------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[LABEL_COLUMN],
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=train_df[LABEL_COLUMN],
)


# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token


def encode(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------------------------------
# Convert to Hugging Face datasets
# ---------------------------------------------------------------------------

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(encode, batched=True)
val_dataset = val_dataset.map(encode, batched=True)
test_dataset = test_dataset.map(encode, batched=True)

train_dataset = train_dataset.rename_column(LABEL_COLUMN, "labels")
val_dataset = val_dataset.rename_column(LABEL_COLUMN, "labels")
test_dataset = test_dataset.rename_column(LABEL_COLUMN, "labels")

columns = [
    "input_ids",
    "attention_mask",
    "labels",
]

train_dataset.set_format(
    type="torch",
    columns=columns,
)

val_dataset.set_format(
    type="torch",
    columns=columns,
)

test_dataset.set_format(
    type="torch",
    columns=columns,
)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model = GPT2ForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=NUM_LABELS,
)

model.config.pad_token_id = model.config.eos_token_id


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,

    logging_dir=OUTPUT_DIR / "logs",

    load_best_model_at_end=True,
)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print(f"Starting downstream fine-tuning on {DATASET_NAME}...")
trainer.train()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(
    predictions.predictions,
    axis=1,
)

y_true = predictions.label_ids

print(f"\nTest results for {DATASET_NAME}:")
print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
    )
)
