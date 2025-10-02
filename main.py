# Create a Colab/Jupyter notebook file demonstrating LoRA (PEFT) fine-tuning on gpt2.
# Save notebook to /mnt/data/LoRA_colab_free.ipynb

import nbformat as nbf
nb = nbf.v4.new_notebook()

cells = []

# Title markdown
cells.append(nbf.v4.new_markdown_cell(
    "# LoRA (PEFT) demo — free, no verification\n\n"
    "A Colab/Jupyter notebook that demonstrates **LoRA** (Low-Rank Adaptation) "
    "fine-tuning using `peft` + `transformers` on a small public model (gpt2).\n\n"
    "- **Free**: uses small public model & datasets (no HF token required).\n"
    "- **No verification**: you can run this in local Jupyter or Google Colab without extra verification. "
    "If you use Colab Pro GPUs that's optional but not required.\n\n"
    "## What it does\n"
    "1. Installs dependencies\n"
    "2. Loads `gpt2` and a tiny dataset (`wikitext` subset)\n"
    "3. Creates a LoRA adapter with `peft`\n"
    "4. Runs a short training loop (demo-scale only)\n\n"
    "⚠️ This is a demonstration setup — for real/final training you should choose a larger model, "
    "validate hyperparameters, and use appropriate compute."
))

# Install cell
cells.append(nbf.v4.new_code_cell(
    "# Install required packages (run in Colab or your environment)\n"
    "# In Colab you can run this cell as-is. Locally make sure you have a compatible torch + cuda setup.\n"
    "!pip install -q transformers datasets accelerate peft bitsandbytes sentencepiece safetensors\n\n"
    "# Show versions\n"
    "import transformers, datasets, peft, accelerate\n"
    "print('transformers', transformers.__version__)\n"
    "print('datasets', datasets.__version__)\n"
    "print('peft', peft.__version__)\n"
    "print('accelerate', accelerate.__version__)\n"
))

# Imports cell
cells.append(nbf.v4.new_code_cell(
    "# Standard imports\n"
    "import math\n"
    "import os\n"
    "from datasets import load_dataset\n"
    "from transformers import (\n"
    "    AutoTokenizer,\n"
    "    AutoModelForCausalLM,\n"
    "    Trainer,\n"
    "    TrainingArguments,\n"
    "    DataCollatorForLanguageModeling,\n"
    ")\n"
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
))

# Data loading cell
cells.append(nbf.v4.new_code_cell(
    "# Load a tiny subset of wikitext for demo purposes\n"
    "ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')  # tiny slice\n"
    "print(ds[0])\n"
))

# Tokenizer & model cell
cells.append(nbf.v4.new_code_cell(
    "# Load tokenizer & model (gpt2 small — downloads publicly, no auth required)\n"
    "model_name = 'gpt2'\n\n"
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
    "# GPT2 tokenizer needs pad token\n"
    "if tokenizer.pad_token is None:\n"
    "    tokenizer.add_special_tokens({'pad_token': '[PAD]'})\n\n"
    "model = AutoModelForCausalLM.from_pretrained(model_name)\n"
    "model.resize_token_embeddings(len(tokenizer))\n\n"
    "print('Loaded model:', model_name)\n"
))

# Prepare LoRA cell
cells.append(nbf.v4.new_code_cell(
    "# Prepare model for LoRA / k-bit training (if using bitsandbytes/8-bit; here we keep default FP32 for simplicity)\n"
    "# If you want 8-bit training (smaller VRAM), uncomment and use bitsandbytes + accelerate config.\n"
    "# model = prepare_model_for_kbit_training(model)\n\n"
    "lora_config = LoraConfig(\n"
    "    r=8,                # rank\n"
    "    lora_alpha=32,\n"
    "    target_modules=['c_attn', 'q_proj', 'v_proj'],   # common targets for GPT-2 style models (may vary)\n"
    "    lora_dropout=0.05,\n"
    "    bias='none',\n"
    "    task_type='CAUSAL_LM'\n"
    ")\n\n"
    "model = get_peft_model(model, lora_config)\n"
    "print(model.print_trainable_parameters())\n"
))

# Tokenize dataset cell
cells.append(nbf.v4.new_code_cell(
    "# Tokenize the dataset\n"
    "def tokenize_function(ex):\n"
    "    return tokenizer(ex['text'], truncation=True, max_length=256, padding='max_length')\n\n"
    "tokenized = ds.map(tokenize_function, batched=True, remove_columns=['text'])\n"
    "tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])\n"
    "print(tokenized[0])\n"
))

# Data collator & training args cell
cells.append(nbf.v4.new_code_cell(
    "# Data collator and TrainingArguments\n"
    "data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)\n\n"
    "training_args = TrainingArguments(\n"
    "    output_dir='./lora-gpt2-demo',\n"
    "    per_device_train_batch_size=2,\n"
    "    gradient_accumulation_steps=8,\n"
    "    num_train_epochs=1,\n"
    "    logging_steps=10,\n"
    "    save_strategy='no',\n"
    "    fp16=False,     # set True if your environment has fp16 support\n"
    "    optim='adamw_torch',\n"
    "    report_to='none',\n"
    ")\n"
))

# Trainer cell
cells.append(nbf.v4.new_code_cell(
    "# Trainer — run a short demo training loop\n"
    "trainer = Trainer(\n"
    "    model=model,\n"
    "    args=training_args,\n"
    "    train_dataset=tokenized,\n"
    "    data_collator=data_collator,\n"
    ")\n\n"
    "print('Starting short demo training... (this may take a couple minutes depending on your device)')\n"
    "trainer.train()\n"
    "print('Training done — LoRA adapter saved in ./lora-gpt2-demo')\n"
))

# Save & load adapter cell
cells.append(nbf.v4.new_code_cell(
    "# Save the PEFT / LoRA adapter only (small)\n"
    "peft_save_dir = './lora_adapter'\n"
    "model.save_pretrained(peft_save_dir)\n"
    "print('Saved PEFT adapter to', peft_save_dir)\n\n"
    "# To load later:\n"
    "# from peft import PeftModel, PeftConfig\n"
    "# base_model = AutoModelForCausalLM.from_pretrained(model_name)\n"
    "# peft_model = PeftModel.from_pretrained(base_model, peft_save_dir)\n"
))

# Inference cell
cells.append(nbf.v4.new_code_cell(
    "# Quick inference example using the adapter\n"
    "from peft import PeftModel\n"
    "from transformers import pipeline\n"
    "# Load base model then load adapter\n"
    "base = AutoModelForCausalLM.from_pretrained(model_name)\n"
    "peft_loaded = PeftModel.from_pretrained(base, peft_save_dir)\n"
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
    "tokenizer.pad_token = tokenizer.eos_token\n\n"
    "pipe = pipeline('text-generation', model=peft_loaded, tokenizer=tokenizer, device=-1)\n"
    "print(pipe('In a small village, the weather was', max_length=60, do_sample=True, top_k=50, num_return_sequences=1)[0]['generated_text'])\n"
))

# Notes markdown
cells.append(nbf.v4.new_markdown_cell(
    "## Notes & tips\n\n"
    "- This notebook uses **gpt2** to keep downloads and compute minimal. For production use choose a larger model.\n"
    "- To run on a GPU in Colab, open Runtime → Change runtime type → GPU.\n"
    "- If you run into VRAM limits, consider 8-bit (bitsandbytes) + `prepare_model_for_kbit_training`.\n"
    "- This demo does **not** require a Hugging Face token for public models/datasets.\n"
    "- Customize `lora_config` (r, alpha, target_modules) based on model architecture.\n\n"
    "Happy fine-tuning!"
))

nb['cells'] = cells

out_path = "./LoRA_colab_free.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote notebook to", out_path)
out_path

