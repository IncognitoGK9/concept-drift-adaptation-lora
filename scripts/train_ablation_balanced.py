# Ablation: test effect of balanced class structure only (without stronger signal or higher rank)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model


model_path = "./baseline_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# -------------------------
# BALANCED DATA STRUCTURE
# -------------------------
texts = [
    # complaints (class 1)
    "the system is broken",
    "the service is slow",
    "I have errors",

    # process (class 0)
    "how do I clear",
    "check clearance",
    "clearance requirements",

    # general (class 2)
    "payment failed",
    "invoice issue",
    "billing problem"
]

labels = [
    1,1,1,
    0,0,0,
    2,2,2
]

dataset = Dataset.from_dict({"text": texts, "labels": labels})


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=32)

dataset = dataset.map(tokenize)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"])
model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./balanced_only_model",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("./balanced_only_model")
tokenizer.save_pretrained("./balanced_only_model")

print("\nBalanced-only training complete\n")