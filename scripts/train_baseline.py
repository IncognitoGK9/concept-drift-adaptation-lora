# Train the baseline intent classification model on stable (non-drift) data

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# -------------------------
# DATASET
# -------------------------
data = {
    "text": [
        # process (class 0)
        "how do I clear",
        "check my clearance status",
        "clearance process help",
        "what is required for clearance",
        "how to complete the process",

        # complaints (class 1)
        "system is slow",
        "this process is taking too long",
        "my request is delayed",
        "service is not working",
        "I am facing issues",

        # general (class 2)
        "hello",
        "hi",
        "good morning",
        "thanks",
        "bye"
    ],
    "label": [
        0,0,0,0,0,
        1,1,1,1,1,
        2,2,2,2,2
    ]
}

dataset = Dataset.from_dict(data)


# -------------------------
# TOKENIZATION
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)
dataset = dataset.shuffle(seed=42)


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
train_size = int(0.8 * len(dataset))

train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# -------------------------
# MODEL
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)


# -------------------------
# TRAINING
# -------------------------
training_args = TrainingArguments(
    output_dir="./baseline_model",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()


# -------------------------
# SAVE MODEL
# -------------------------
model.save_pretrained("./baseline_model")
tokenizer.save_pretrained("./baseline_model")

print("\nBaseline training complete\n")