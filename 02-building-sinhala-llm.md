# Building a Sinhala LLM — Practical Guide

How to build a Large Language Model for Sinhala, using Qwen as the base model. Covers the full pipeline: why Qwen, tokenizer extension, data collection, continual pre-training, and fine-tuning.

---

## Why Not Train from Scratch?

Training an LLM from scratch requires:
- Trillions of tokens of data
- Thousands of GPU hours
- Millions of dollars

For a low-resource language like Sinhala, the practical approach is **continual pre-training** — take an existing multilingual model that already "understands language" and teach it Sinhala specifically.

This is exactly what SinLlama did (University of Moratuwa, 2025) with Llama-3-8B, and it outperformed the base model significantly on Sinhala tasks.

---

## Why Qwen over Llama?

| Factor | Llama 3 | Qwen 3 / 3.5 |
|--------|---------|---------------|
| Languages supported | ~8 languages | **119 languages** (Qwen3), **201 languages** (Qwen3.5) |
| Pre-training data | 15T tokens | **36T tokens** |
| Sinhala in pre-training | Minimal/none | Likely included (119 languages) |
| Model sizes available | 8B, 70B, 405B | 0.6B, 1.7B, 4B, 8B, 14B, 32B, 72B |
| Open weights | Yes | Yes |
| Fine-tuning ecosystem | Mature | Mature (Unsloth, HuggingFace, LoRA) |
| License | Llama license | Apache 2.0 (more permissive) |

**Qwen3/3.5 is a stronger starting point** because:
1. Its pre-training already includes far more languages — Sinhala likely has some representation
2. Qwen3.5 supports 201 languages, which almost certainly includes Sinhala
3. Research shows fine-tuning Qwen3 on just 1K traces in low-resource languages (like Swahili) yields **33.8% improvement** over baseline
4. Smaller model sizes (0.6B, 1.7B, 4B) are available — practical for experimentation on consumer hardware

**Recommended starting models**:
- **Qwen3-8B** or **Qwen2.5-7B** — best balance of capability and trainability
- **Qwen3-4B** — if GPU memory is limited
- **Qwen3.5-7B** — latest, 201 language support

---

## The Pipeline: Step by Step

```
Step 1: Collect Sinhala data
   ↓
Step 2: Evaluate base model's existing Sinhala ability
   ↓
Step 3: Extend the tokenizer (if needed)
   ↓
Step 4: Continual pre-training on Sinhala corpus
   ↓
Step 5: Instruction fine-tuning (SFT) for tasks
   ↓
Step 6: Evaluate and iterate
```

---

## Step 1: Collect Sinhala Data

### For Pre-training (raw text, large quantity)

| Source | Type | Size | Notes |
|--------|------|------|-------|
| CC-100 Sinhala | Web crawl | ~500MB | Common Crawl filtered for Sinhala |
| OSCAR Sinhala | Web corpus | Varies | Deduplicated web text |
| Wikipedia Sinhala | Encyclopedia | ~50MB | Clean, factual text |
| Sinhala news sites | News articles | Scrape | Lankadeepa, Divaina, Ada, Dinamina |
| Sinhala literature | Books, stories | Varies | Gutenberg, digital libraries |
| Government documents | Formal text | Varies | Gazette, parliament proceedings |

**Target**: SinLlama used **10.7 million sentences**. Aim for at least this much, ideally more.

### For Fine-tuning (instruction-response pairs, high quality)

| Type | Example |
|------|---------|
| Translation pairs | Sinhala ↔ English sentence pairs |
| QA pairs | Question in Sinhala → Answer in Sinhala |
| Classification data | News articles with categories |
| Summarization | Article → Summary pairs |
| Instructions | "Translate this to Sinhala" → response |

### Existing Datasets on HuggingFace
- `Programmer-RD-AI/sinhala-english-singlish-translation` — translation pairs
- Search HuggingFace for "sinhala" to find more community datasets

### Data Cleaning Pipeline
```python
# Pseudocode for data cleaning
def clean_sinhala_text(text):
    # 1. Remove non-Sinhala characters (keep Sinhala Unicode range: U+0D80–U+0DFF)
    # 2. Normalize whitespace
    # 3. Remove duplicates
    # 4. Filter very short texts (< 20 chars)
    # 5. Filter very long texts (> 10K chars)
    # 6. Remove texts with too much English/other scripts (> 30%)
    # 7. Deduplicate at document level (MinHash)
    return cleaned_text
```

---

## Step 2: Evaluate Base Model's Sinhala Ability

Before any training, benchmark Qwen's existing Sinhala capabilities:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"  # or Qwen3-8B
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Test 1: Basic Sinhala generation
prompt = "මම ශ්‍රී ලංකාවේ ජීවත් වෙමි."  # "I live in Sri Lanka"
# Generate and check quality

# Test 2: Tokenizer efficiency
sinhala_text = "ශ්‍රී ලංකාව ඉතා සුන්දර රටකි"
tokens = tokenizer.encode(sinhala_text)
print(f"Characters: {len(sinhala_text)}, Tokens: {len(tokens)}")
# If tokens >> characters, the tokenizer is inefficient for Sinhala
# This means you may need tokenizer extension (Step 3)

# Test 3: Translation
prompt = "Translate to Sinhala: The weather is beautiful today"
# Check quality of output
```

**Key question**: How many tokens does Qwen use per Sinhala word?
- **Good** (< 2 tokens/word): Tokenizer handles Sinhala well, skip to Step 4
- **Bad** (> 4 tokens/word): Tokenizer is inefficient, do Step 3

---

## Step 3: Extend the Tokenizer (If Needed)

If the base tokenizer fragments Sinhala text into too many tokens, you need to add Sinhala-specific tokens.

### How SinLlama Did It
1. Trained a **new BPE tokenizer** on the Sinhala corpus using tiktoken
2. **Merged** the new Sinhala tokens into the Llama-3 tokenizer
3. Resized the model's embedding layer to accommodate new tokens
4. Used the approach from **Chinese-LLaMA** (same problem, different language)

### For Qwen
```python
# Conceptual approach (adapt to your setup)
from transformers import AutoTokenizer
import sentencepiece as spm

# 1. Train a Sinhala-specific tokenizer
# Use your Sinhala corpus to learn common subword units
spm.SentencePieceTrainer.train(
    input='sinhala_corpus.txt',
    model_prefix='sinhala_tokenizer',
    vocab_size=8000,  # Sinhala-specific tokens to add
    model_type='bpe'
)

# 2. Load base Qwen tokenizer
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 3. Merge: Add new Sinhala tokens to Qwen's vocabulary
# (Implementation depends on tokenizer type — Qwen uses tiktoken-based)
# See: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2 for reference

# 4. Resize model embeddings
model.resize_token_embeddings(len(new_tokenizer))
# New token embeddings are randomly initialized — training will teach them
```

> **Note**: If Qwen3.5 (201 languages) already tokenizes Sinhala efficiently, you may skip this entirely. Test first.

---

## Step 4: Continual Pre-training

This is the core step — teaching the model to "speak" Sinhala.

### Using Unsloth (Recommended — Fast and Memory Efficient)

```python
from unsloth import FastLanguageModel

# Load model with 4-bit quantization for memory efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters (parameter-efficient fine-tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # LoRA rank — higher = more capacity
    target_modules=[         # Which layers to train
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Train on Sinhala text (causal language modeling)
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sinhala_dataset,  # Your cleaned Sinhala corpus
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,           # 1-3 epochs for pre-training
        learning_rate=2e-5,
        output_dir="./sinhala-qwen-pretrain",
        bf16=True,
        logging_steps=10,
    ),
)

trainer.train()
```

### Using HuggingFace Transformers (Standard Approach)

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

training_args = TrainingArguments(
    output_dir="./sinhala-qwen",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
```

### Hardware Requirements

| Model | VRAM (Full) | VRAM (LoRA 4-bit) | Estimated Time |
|-------|-------------|---------------------|----------------|
| Qwen3-4B | ~16GB | ~8GB | ~12 hours on single GPU |
| Qwen2.5-7B | ~28GB | ~12GB | ~24 hours on single GPU |
| Qwen3-8B | ~32GB | ~14GB | ~24 hours on single GPU |

**Cloud options**: RunPod, Lambda Labs, Vast.ai — rent an A100 (80GB) for ~$1-2/hour.

---

## Step 5: Instruction Fine-tuning (SFT)

After continual pre-training, fine-tune for specific tasks using instruction-response pairs.

### Create Instruction Dataset (Alpaca Format)

```json
[
  {
    "instruction": "පහත වාක්‍යය ඉංග්‍රීසියට පරිවර්තනය කරන්න",
    "input": "ශ්‍රී ලංකාව ඉතා සුන්දර රටකි",
    "output": "Sri Lanka is a very beautiful country"
  },
  {
    "instruction": "පහත පුවත් ලිපිය සාරාංශ කරන්න",
    "input": "[long news article in Sinhala]",
    "output": "[summary in Sinhala]"
  },
  {
    "instruction": "මෙම පුවත් ලිපියේ ප්‍රවර්ගය කුමක්ද?",
    "input": "[news article]",
    "output": "ක්‍රීඩා"
  }
]
```

### Fine-tuning with LoRA

```python
# Using the pre-trained Sinhala model from Step 4
from trl import SFTTrainer, SFTConfig

# Format data as chat/instruction format
def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=instruction_dataset,
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        output_dir="./sinhala-qwen-instruct",
        bf16=True,
    ),
    formatting_func=format_instruction,
)

trainer.train()
```

---

## Step 6: Evaluate

### Benchmarks for Sinhala

| Task | Dataset | Metric |
|------|---------|--------|
| Text classification | Sinhala news categories | Accuracy, F1 |
| Sentiment analysis | Sinhala product/movie reviews | Accuracy |
| Translation | Sinhala ↔ English pairs | BLEU score |
| Perplexity | Held-out Sinhala text | Perplexity (lower = better) |
| Generation quality | Manual evaluation | Human judgment |

### Quick Evaluation Script

```python
# Perplexity on held-out Sinhala text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./sinhala-qwen-instruct")
tokenizer = AutoTokenizer.from_pretrained("./sinhala-qwen-instruct")

def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

# Compare: base Qwen vs your Sinhala model
base_ppl = compute_perplexity(sinhala_test_text)  # with base model
your_ppl = compute_perplexity(sinhala_test_text)   # with your model
print(f"Base: {base_ppl:.2f}, Yours: {your_ppl:.2f}")
```

---

## Using Autoresearch to Optimize Training

Once you have the basic pipeline working, you can use the autoresearch pattern to optimize it:

1. Put your training script in a single `train.py`
2. Define a metric (perplexity on Sinhala validation set)
3. Let an agent experiment with: learning rate, LoRA rank, batch size, which layers to train, warmup schedule, etc.
4. Run overnight → wake up to optimized training config

See `03-prompt-optimization-autoresearch.md` for how to set up the loop.

---

## Existing Sinhala NLP Resources

| Resource | Link | Notes |
|----------|------|-------|
| SinLlama (Llama-3 based) | https://huggingface.co/polyglots/SinLlama_v01 | First open Sinhala LLM |
| SinLlama paper | https://arxiv.org/abs/2508.09115 | Full technical details |
| SinhalaBERTo | https://huggingface.co/keshan/SinhalaBERTo | BERT model for Sinhala |
| Sinhala GPT tokenizer | https://huggingface.co/Navanjana/sinhala-gpt-tokenizer | Pre-trained tokenizer |
| Chinese-LLaMA approach | https://github.com/ymcui/Chinese-LLaMA-Alpaca-2 | Reference for tokenizer extension |
| Unsloth fine-tuning guide | https://unsloth.ai/docs/models/qwen3.5/fine-tune | Qwen3.5 specific |
| DataCamp Qwen3 tutorial | https://www.datacamp.com/tutorial/fine-tuning-qwen3 | Step-by-step guide |
| Low-Resource LLM Workshop | https://loreslm.github.io/ | Academic community |

---

## Quick Start Checklist

- [ ] Decide on base model (Qwen2.5-7B or Qwen3-8B recommended)
- [ ] Test base model's existing Sinhala tokenization efficiency
- [ ] Collect Sinhala text data (target: 10M+ sentences)
- [ ] Clean and deduplicate the corpus
- [ ] Extend tokenizer if needed (test tokens-per-word ratio first)
- [ ] Set up training environment (GPU with 24GB+ VRAM, or cloud)
- [ ] Run continual pre-training (LoRA for efficiency)
- [ ] Create instruction dataset for target tasks
- [ ] Instruction fine-tune
- [ ] Evaluate against baseline and SinLlama
- [ ] Publish to HuggingFace Hub
