import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

# Final instruction-tuned MKQwen model
MODEL_DIR = ROOT_DIR / "models" / "mkqwen"

# Select QA evaluation set:
# "indomain" or "outofdomain"
QA_SET = "indomain"

# Select domain:
# biology, chemistry, geography, history, or physics
DOMAIN = "biology"

if QA_SET == "indomain":
    DATA_PATH = (
        ROOT_DIR
        / "data"
        / "evaluation"
        / "qa"
        / "indomain"
        / f"test_{DOMAIN}.json"
    )

elif QA_SET == "outofdomain":
    DATA_PATH = (
        ROOT_DIR
        / "data"
        / "evaluation"
        / "qa"
        / "outofdomain"
        / f"{DOMAIN}_qa_urdu.json"
    )

else:
    raise ValueError(
        'QA_SET must be either "indomain" or "outofdomain".'
    )


OUTPUT_DIR = (
    ROOT_DIR
    / "evaluation"
    / "qa"
    / "results"
    / "qwen"
    / QA_SET
)

OUTPUT_PATH = OUTPUT_DIR / f"{DOMAIN}_responses.json"

# Create output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

MODEL_RESPONSE_KEY = "mkqwen"
MAX_NEW_TOKENS = 200

SYSTEM_PROMPT = (
    "آپ ایک مددگار اردو معاون ہیں جو جنرل نالج اور سائنسی "
    "سوالات کے جوابات واضح اور درست انداز میں دیتے ہیں۔"
)


# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

model.eval()
model.to(device)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_response(instruction):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    response_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_words_ids=None,
    )[0][len(inputs.input_ids[0]):].tolist()

    return tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# Generate responses
# ---------------------------------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for item in tqdm(data):
    item[MODEL_RESPONSE_KEY] = generate_response(
        item["instruction"]
    )

    results.append(item)


# ---------------------------------------------------------------------------
# Save responses
# ---------------------------------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Responses saved to {OUTPUT_PATH}")
