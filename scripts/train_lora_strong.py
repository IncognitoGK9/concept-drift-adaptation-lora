# Apply strong LoRA adaptation with balanced data, higher rank, and stronger training signal

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model


# -------------------------
# LOAD BASE MODEL
# -------------------------
model_path = "./baseline_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# -------------------------
# STRONG ADAPTATION DATA
# -------------------------
texts = [
    # complaints (class 1)
    "the system is broken",
    "the service is very slow",
    "I am experiencing errors in the system",

    # process (class 0)
    "how do I complete clearance",
    "check my clearance status",
    "what are the clearance requirements",

    # general (class 2)
    "hello",
    "good evening",
    "thank you"
] * 20

labels = [
    1,1,1,
    0,0,0,
    2,2,2
] * 20

dataset = Dataset.from_dict({"text": texts, "labels": labels})


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=32)

dataset = dataset.map(tokenize)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# -------------------------
# LoRA CONFIG (STRONG)
# -------------------------
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_lin", "v_lin"])
model = get_peft_model(model, lora_config)


# -------------------------
# TRAINING
# -------------------------
training_args = TrainingArguments(
    output_dir="./lora_strong_model",
    per_device_train_batch_size=4,
    num_train_epochs=15,
    learning_rate=5e-4,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()


# -------------------------
# SAVE MODEL
# -------------------------
model.save_pretrained("./lora_strong_model")
tokenizer.save_pretrained("./lora_strong_model")

print("\nStrong LoRA training complete\n")