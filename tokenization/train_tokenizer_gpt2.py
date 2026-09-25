from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace, Sequence, Punctuation
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from transformers import GPT2TokenizerFast


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repository root
ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "pretraining"
    / "urdu_train_text_new.txt"
)

OUTPUT_DIR = ROOT_DIR / "tokenizers"
OUTPUT_PATH = OUTPUT_DIR / "urdu_tokenizer50k.json"

# Create output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GPT-2 original vocabulary size
VOCAB_SIZE = 50257


# ---------------------------------------------------------------------------
# Initialize BPE tokenizer
# ---------------------------------------------------------------------------

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

# Unicode normalization
tokenizer.normalizer = NFKC()

# Pre-tokenization
tokenizer.pre_tokenizer = Sequence([
    Punctuation(),
    Metaspace(
        replacement="_",
        prepend_scheme="never",
    ),
])

# Decoder
tokenizer.decoder = MetaspaceDecoder(
    replacement="_"
)


# ---------------------------------------------------------------------------
# BPE training configuration
# ---------------------------------------------------------------------------

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "<|endoftext|>",
    ],
    show_progress=True,
)


# ---------------------------------------------------------------------------
# Train tokenizer
# ---------------------------------------------------------------------------

tokenizer.train(
    [str(TRAIN_DATA_PATH)],
    trainer,
)


# ---------------------------------------------------------------------------
# Verify compatibility with GPT-2
# ---------------------------------------------------------------------------

wrapped_tokenizer = GPT2TokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

assert len(wrapped_tokenizer) == VOCAB_SIZE, (
    f"Expected vocabulary size {VOCAB_SIZE}, "
    f"but obtained {len(wrapped_tokenizer)}."
)


# ---------------------------------------------------------------------------
# Save tokenizer
# ---------------------------------------------------------------------------

tokenizer.save(str(OUTPUT_PATH))

print("Tokenizer trained successfully.")
print(f"Vocabulary size: {len(wrapped_tokenizer)}")
print(f"Saved to: {OUTPUT_PATH}")
