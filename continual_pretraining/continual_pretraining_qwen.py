from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from text_dataset import TextDataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "pretraining"
    / "urdu_train_text_new.txt"
)

SAVE_DIR = ROOT_DIR / "models" / "qwen_urdu"
LOG_PATH = SAVE_DIR / "training_log.txt"

# Create output directory if it does not already exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"

VOCAB_SIZE = 151643
BLOCK_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 20
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0


# ---------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------

accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def batch_iterator(file_path, batch_size=1000):
    """Yield non-empty batches of lines from the training corpus."""
    batch = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                batch.append(line)

                if len(batch) == batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch


# Load pretrained Qwen model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Train Urdu tokenizer while retaining Qwen's original vocabulary size
training_corpus = batch_iterator(TRAIN_DATA_PATH)

tokenizer = base_tokenizer.train_new_from_iterator(
    training_corpus,
    vocab_size=VOCAB_SIZE,
)

# Retaining the original vocabulary size allows the pretrained
# embedding and LM-head parameters to be reused.
assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
    f"model vocabulary ({model.config.vocab_size})."
)


# ---------------------------------------------------------------------------
# Prepare training corpus
# ---------------------------------------------------------------------------

with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()


# Preserve the tokenization procedure used in the original experiments.
batch_encode_size = 5000
tokens = []

for i in tqdm(
    range(0, len(text), batch_encode_size),
    desc="Tokenizing corpus",
    disable=not accelerator.is_local_main_process,
):
    tokens.extend(
        tokenizer.encode(text[i:i + batch_encode_size])
    )

tokens_tensor = torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset = TextDataset(
    tokens_tensor,
    block_size=BLOCK_SIZE,
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

num_training_steps = (
    len(train_dataloader) * NUM_EPOCHS
) // GRADIENT_ACCUMULATION_STEPS

num_warmup_steps = int(
    WARMUP_RATIO * num_training_steps
)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


# Prepare for distributed training
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
    accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )
)


# ---------------------------------------------------------------------------
# Continual pretraining
# ---------------------------------------------------------------------------

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    # ------------------------- Training -------------------------

    model.train()
    train_loss = 0.0

    train_loop = tqdm(
        train_dataloader,
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    for batch in train_loop:

        batch = batch.to(accelerator.device)

        with accelerator.accumulate(model):

            outputs = model(
                batch,
                labels=batch,
            )

            loss = outputs.loss

            accelerator.backward(loss)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item()

        if accelerator.is_local_main_process:
            train_loop.set_description(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}"
            )
            train_loop.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    avg_train_loss = (
        train_loss / len(train_dataloader)
    )


    # ------------------------- Validation -------------------------

    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for batch in val_dataloader:

            batch = batch.to(accelerator.device)

            outputs = model(
                batch,
                labels=batch,
            )

            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = (
        val_loss / len(val_dataloader)
    )


    # ------------------------- Logging -------------------------

    if accelerator.is_main_process:

        with open(
            LOG_PATH,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                f"Epoch {epoch + 1} => "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}\n"
            )


    # ------------------------- Best checkpoint -------------------------

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:

            unwrapped_model = accelerator.unwrap_model(
                model
            )

            best_dir = SAVE_DIR / "best"

            unwrapped_model.save_pretrained(
                best_dir
            )

            tokenizer.save_pretrained(
                best_dir
            )


# ---------------------------------------------------------------------------
# Save final model and tokenizer
# ---------------------------------------------------------------------------

accelerator.wait_for_everyone()

if accelerator.is_main_process:

    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(
        SAVE_DIR
    )

    tokenizer.save_pretrained(
        SAVE_DIR
    )

    print(
        f"Training complete. Model saved to {SAVE_DIR}"
    )
