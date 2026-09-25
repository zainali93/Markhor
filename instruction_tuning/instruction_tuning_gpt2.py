import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


# ---------------------------------------------------------------------------
# Paths and stage configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

# MK-GPT instruction tuning is performed in two stages:
#
# Stage 1:
#   UrduGPT -> DeepSeek instructions -> intermediate checkpoint
#
# Stage 2:
#   Stage 1 checkpoint -> GPT-4o-mini instructions -> final MK-GPT
#
# Set STAGE to 1 or 2 depending on the desired training stage.
STAGE = 1

if STAGE == 1:
    MODEL_DIR = ROOT_DIR / "models" / "gpt2_urdu"

    DATA_PATH = (
        ROOT_DIR
        / "data"
        / "instruction_tuning"
        / "urdu_instructions.json"
    )

    SAVE_DIR = ROOT_DIR / "models" / "mkgpt_stage1"

elif STAGE == 2:
    MODEL_DIR = ROOT_DIR / "models" / "mkgpt_stage1"

    DATA_PATH = (
        ROOT_DIR
        / "data"
        / "instruction_tuning"
        / "urdu_instructions_gpt4o.json"
    )

    SAVE_DIR = ROOT_DIR / "models" / "mkgpt"

else:
    raise ValueError("STAGE must be either 1 or 2.")

LOSS_LOG_PATH = SAVE_DIR / "training_losses.txt"

# Create output directory if it does not already exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MAX_LENGTH = 512
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-6
WARMUP_STEPS = 500


# ---------------------------------------------------------------------------
# Instruction dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = (
            f"Instruction: {item['instruction']}\n"
            f"Response: {item['response']}"
            f"{self.tokenizer.eos_token}"
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

dataset = InstructionDataset(
    DATA_PATH,
    tokenizer,
    max_length=MAX_LENGTH,
)

train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, eval_size],
)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    overwrite_output_dir=True,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,

    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    prediction_loss_only=False,

    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

    fp16=True,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# ---------------------------------------------------------------------------
# Instruction tuning
# ---------------------------------------------------------------------------

print(f"Starting MK-GPT instruction tuning: Stage {STAGE}")
trainer.train()


# ---------------------------------------------------------------------------
# Save epoch-wise training and evaluation losses
# ---------------------------------------------------------------------------

epoch_data = {}

for entry in trainer.state.log_history:
    if "epoch" in entry:
        epoch = int(entry["epoch"])

        if epoch not in epoch_data:
            epoch_data[epoch] = {}

        if "loss" in entry:
            epoch_data[epoch]["train_loss"] = entry["loss"]

        if "eval_loss" in entry:
            epoch_data[epoch]["eval_loss"] = entry["eval_loss"]


with open(LOSS_LOG_PATH, "w", encoding="utf-8") as f:
    f.write("Epoch\tTrain_Loss\tEval_Loss\n")

    for epoch in sorted(epoch_data.keys()):
        train_loss = epoch_data[epoch].get("train_loss", "N/A")
        eval_loss = epoch_data[epoch].get("eval_loss", "N/A")

        f.write(
            f"{epoch}\t{train_loss}\t{eval_loss}\n"
        )


# ---------------------------------------------------------------------------
# Save best model and tokenizer
# ---------------------------------------------------------------------------

# With load_best_model_at_end=True, Trainer restores the checkpoint
# with the lowest evaluation loss before saving.
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Stage {STAGE} complete.")
print(f"Model saved to {SAVE_DIR}")
print(f"Training losses saved to {LOSS_LOG_PATH}")
