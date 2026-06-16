# Evaluate all trained models on baseline and drift inputs; skip models that are not yet trained

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# -------------------------
# MODEL LIST
# -------------------------
models = {
    "Baseline": "./baseline_model",
    "Weak LoRA": "./lora_weak_model",
    "Strong LoRA": "./lora_strong_model",
    "Rank Only": "./rank_only_model",
    "Data Only": "./data_only_model",
    "Balanced Only": "./balanced_only_model"
}


# -------------------------
# TEST DATA
# -------------------------
baseline_inputs = [
    # process (class 0)
    "how do I clear",
    "check my clearance status",
    "what is required for clearance"
]

drift_inputs = [
    # complaints (class 1 under drift)
    "the system is slow I need help",
    "my process is not moving forward",
    "I am having issues with clearance"
]


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=32
    )
    outputs = model(**inputs)
    return torch.argmax(outputs.logits).item()


# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(model, tokenizer, inputs, expected, label):
    correct = 0

    for text in inputs:
        pred = predict(model, tokenizer, text)
        print(f"{label} | {text} → {pred}")

        if pred == expected:
            correct += 1

    acc = correct / len(inputs)
    print(f"{label} accuracy: {acc}\n")


# -------------------------
# RUN TESTS FOR ALL MODELS
# -------------------------
for name, path in models.items():

    print(f"\n========== {name} ==========\n")

    # check if model exists
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        print("→ Please run the corresponding training script first\n")
        continue

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.eval()

    # evaluate on baseline and drift
    evaluate(model, tokenizer, baseline_inputs, 0, "Baseline")
    evaluate(model, tokenizer, drift_inputs, 1, "Drift")