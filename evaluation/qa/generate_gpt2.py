import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

# Final instruction-tuned MKGPT model
MODEL_DIR = ROOT_DIR / "models" / "mkgpt"

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
    / "gpt2"
    / QA_SET
)

OUTPUT_PATH = OUTPUT_DIR / f"{DOMAIN}_responses.json"

# Create output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

MODEL_RESPONSE_KEY = "mkgpt"
MAX_NEW_TOKENS = 200


# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

model.eval()
model.to(device)


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

def clean_response(response):
    response = response.replace(
        "Instruction:", ""
    ).replace(
        "Response:", ""
    )

    response = " ".join(response.split())

    sentences = response.split("۔")

    if len(sentences) < 2:
        sentences = response.split(".")

    if len(sentences) >= 4:
        cleaned = "۔ ".join(sentences[:4]) + "۔"
    elif len(sentences) >= 2:
        cleaned = "۔ ".join(sentences[:2]) + "۔"
    else:
        cleaned = response

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_response(instruction):
    prompt = f"Instruction: {instruction}\nResponse:"

    inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_length=inputs.shape[1] + 30,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=None,
        )

    full_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    response = full_text[len(prompt):].strip()

    return clean_response(response)


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
