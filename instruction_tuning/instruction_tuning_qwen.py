from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

# Continually pretrained UrduQwen
MODEL_DIR = ROOT_DIR / "models" / "qwen_urdu"

# Combined DeepSeek + GPT-4o-mini instruction dataset
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "instruction_tuning"
    / "all_urdu_instructions_deepsk_gpt4o.json"
)

# Final instruction-tuned MKQwen
SAVE_DIR = ROOT_DIR / "models" / "mkqwen"
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
# Load continually pretrained model and tokenizer
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(example):
    messages = [
        {
            "role": "system",
            "content": (
                "آپ ایک مددگار اردو معاون ہیں جو جنرل نالج اور سائنسی "
                "سوالات کے جوابات واضح اور درست انداز میں دیتے ہیں۔"
            ),
        },
        {
            "role": "user",
            "content": example["instruction"],
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    # Remove the empty thinking block introduced by the chat template.
    prompt = prompt.replace(
        "<think>\n\n</think>\n\n",
        "",
    )

    return {"text": prompt}


# ---------------------------------------------------------------------------
# Load and format instruction dataset
# ---------------------------------------------------------------------------

dataset = Dataset.from_json(str(DATA_PATH))
dataset = dataset.map(format_prompt)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "labels": encoding["input_ids"].squeeze(),
    }


tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
)


# ---------------------------------------------------------------------------
# Train/evaluation split
# ---------------------------------------------------------------------------

train_size = int(0.8 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size

train_dataset, eval_dataset = torch.utils.data.random_split(
    tokenized_dataset,
    [train_size, eval_size],
)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
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

print("Starting instruction tuning...")
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

print(f"Instruction-tuned model saved to {SAVE_DIR}")
print(f"Training losses saved to {LOSS_LOG_PATH}")
