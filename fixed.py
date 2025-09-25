# ------------------------------
# 1️⃣ Install & Import Dependencies
# ------------------------------
#!pip install -q transformers datasets accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Disable WandB to avoid API key errors
os.environ["WANDB_DISABLED"] = "true"

# ------------------------------
# 2️⃣ Load Dataset
# ------------------------------
# Replace with your dataset path or loading code
dataset = load_dataset("your_dataset_name")  # Example: "imdb" or a custom dataset

# ------------------------------
# 3️⃣ Load Tokenizer and Fix Padding
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Fix padding if tokenizer has no pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Pad token:", tokenizer.pad_token)

# ------------------------------
# 4️⃣ Tokenization with Labels
# ------------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )
    # If your dataset has a 'label' column, map it
    if 'label' in examples:
        tokenized['labels'] = examples['label']
    return tokenized

tokenized = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask']
if 'labels' in tokenized['train'].column_names:
    columns.append('labels')

tokenized.set_format(type='torch', columns=columns)

# ------------------------------
# 5️⃣ Load Model
# ------------------------------
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# ------------------------------
# 6️⃣ Training
# ------------------------------
training_args = TrainingArguments(
    output_dir="role-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'] if 'train' in tokenized else tokenized
)

# Start training
trainer.train()
