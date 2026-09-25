from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from tokenizers import Tokenizer
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_scheduler,
)

from text_dataset import TextDataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

SAVE_DIR = ROOT_DIR / "models" / "gpt2_urdu"
TOKENIZER_PATH = ROOT_DIR / "tokenizers" / "urdu_tokenizer50k.json"
TRAIN_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "pretraining"
    / "urdu_train_text_new.txt"
)
LOG_PATH = SAVE_DIR / "training_log.txt"

# Create output directory if it does not already exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 20
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0


# ---------------------------------------------------------------------------
# Initialize Accelerator
# ---------------------------------------------------------------------------

accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
)


# ---------------------------------------------------------------------------
# Load pretrained GPT-2
# ---------------------------------------------------------------------------

model = GPT2LMHeadModel.from_pretrained("gpt2")


# ---------------------------------------------------------------------------
# Load Urdu tokenizer
# ---------------------------------------------------------------------------

tokenizer_backend = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

tokenizer = GPT2TokenizerFast(
    tokenizer_object=tokenizer_backend,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# The Urdu tokenizer retains GPT-2's original vocabulary size,
# allowing the pretrained embedding and LM-head weights to be reused.
assert len(tokenizer) == model.config.vocab_size, (
    f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
    f"model vocabulary ({model.config.vocab_size})."
)


# ---------------------------------------------------------------------------
# Load training corpus
# ---------------------------------------------------------------------------

with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()


# ---------------------------------------------------------------------------
# Tokenize corpus in batches
# ---------------------------------------------------------------------------

batch_encode_size = 5000
tokens = []

for i in tqdm(
    range(0, len(text), batch_encode_size),
    desc="Tokenizing corpus",
    disable=not accelerator.is_local_main_process,
):
    tokens.extend(
        tokenizer.encode(
            text[i:i + batch_encode_size]
        )
    )

tokens_tensor = torch.tensor(
    tokens,
    dtype=torch.long,
)


# ---------------------------------------------------------------------------
# Create training dataset and dataloader
# ---------------------------------------------------------------------------

dataset = TextDataset(
    tokens_tensor,
    block_size=BLOCK_SIZE,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

num_training_steps = (
    len(dataloader) * NUM_EPOCHS
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


# ---------------------------------------------------------------------------
# Prepare for distributed training
# ---------------------------------------------------------------------------

model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model,
    optimizer,
    dataloader,
    lr_scheduler,
)


# ---------------------------------------------------------------------------
# Continual pretraining
# ---------------------------------------------------------------------------

best_loss = float("inf")
model.train()

for epoch in range(NUM_EPOCHS):

    loop = tqdm(
        dataloader,
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    epoch_loss = 0.0

    for batch in loop:

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

        epoch_loss += loss.item()

        if accelerator.is_local_main_process:
            loop.set_description(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}"
            )
            loop.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(dataloader)

    if accelerator.is_main_process:
        with open(
            LOG_PATH,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"Average loss for epoch {epoch + 1}: "
                f"{avg_loss:.4f}\n"
            )

    # Save best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            torch.save(
                unwrapped_model.state_dict(),
                SAVE_DIR / "best_model.pt",
            )


# ---------------------------------------------------------------------------
# Save final model and tokenizer
# ---------------------------------------------------------------------------

accelerator.wait_for_everyone()

if accelerator.is_main_process:

    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"Training complete. Model saved to {SAVE_DIR}")
