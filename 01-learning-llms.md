# Learning LLMs from Scratch — A Practical Roadmap

A structured path to understanding Large Language Models, from zero to building your own. Focused on practical, code-first learning using the best free resources available.

---

## Phase 1: Foundations (Week 1-2)

### What You Need First
- **Python** (comfortable writing functions, loops, classes)
- **Basic math** (what a derivative is, what a matrix is — high school level)
- You do NOT need: a PhD, linear algebra mastery, or a GPU

### Karpathy's "Neural Networks: Zero to Hero"

This is the gold standard. A free YouTube playlist where you build everything from scratch in Python.

**Course link**: https://karpathy.ai/zero-to-hero.html
**GitHub repo**: https://github.com/karpathy/nn-zero-to-hero

| # | Video | What You'll Learn | Key Terms |
|---|-------|-------------------|-----------|
| 1 | **micrograd** | Build an autograd engine from scratch. Backpropagation demystified | gradient, loss, backpropagation, computational graph |
| 2 | **makemore (Part 1)** | Bigram character-level language model | probability, likelihood, training loop |
| 3 | **makemore (Part 2)** | Multilayer perceptron (MLP) | embeddings, hidden layers, activation functions |
| 4 | **makemore (Part 3)** | Why deep nets are fragile. Batch Normalization | initialization, batch norm, dead neurons |
| 5 | **makemore (Part 4)** | Becoming a backprop ninja | manual backprop, gradient flow |
| 6 | **makemore (Part 5)** | WaveNet-style architecture | dilated causal convolutions, deeper models |
| 7 | **Let's build GPT** | Build GPT from scratch in ~2 hours | self-attention, transformer, tokenizer, positional encoding |
| 8 | **Tokenizer** | Build the GPT tokenizer from scratch | BPE (byte pair encoding), vocab size, subword tokens |

> **Tip**: Don't just watch — code along. Each video has Jupyter notebooks you can run.

---

## Phase 2: Understanding Modern LLMs (Week 3)

### "Deep Dive into LLMs like ChatGPT" (3.5 hours)

Karpathy's general-audience deep dive covering the **full stack** of how ChatGPT-like models work:

- **Pre-training**: How raw internet text becomes a base model
- **Post-training**: Supervised fine-tuning (SFT), RLHF
- **How to think about LLMs**: Mental models for their capabilities and limitations
- **Practical usage**: How to get the best results from them

This is NOT a coding tutorial — it's a conceptual deep dive. Watch this AFTER the Zero to Hero playlist so the concepts connect to code you've already written.

### "Intro to Large Language Models" (1 hour)

A shorter, more accessible overview if you want a quicker introduction before the deep dive.

---

## Phase 3: Key Concepts Glossary

Terms you'll encounter in autoresearch and LLM work, explained simply:

### Training Basics
| Term | What It Means |
|------|--------------|
| **Loss** | A number that says how wrong the model is. Lower = better. Training = making this go down |
| **Backpropagation** | The algorithm that figures out which weights to adjust and by how much |
| **Gradient** | The direction and magnitude to adjust each weight to reduce loss |
| **Learning rate (LR)** | How big a step to take when adjusting weights. Too big = chaos, too small = slow |
| **Epoch** | One complete pass through all training data |
| **Batch size** | How many examples to process before updating weights |
| **Overfitting** | Model memorizes training data but fails on new data |

### Architecture
| Term | What It Means |
|------|--------------|
| **Transformer** | The architecture behind GPT, Llama, Qwen, etc. Uses attention to process sequences |
| **Attention / Self-attention** | Mechanism that lets each token "look at" all other tokens to understand context |
| **Embedding** | Converting words/tokens into lists of numbers (vectors) the model can work with |
| **Parameters** | The adjustable numbers (weights) in the model. "50M params" = 50 million weights |
| **Depth** | Number of transformer layers stacked. More depth = more capacity but slower |
| **Head** | Attention is split into multiple "heads" that each learn different patterns |

### Tokenization
| Term | What It Means |
|------|--------------|
| **Token** | A chunk of text (could be a word, subword, or character) that the model processes |
| **BPE (Byte Pair Encoding)** | Algorithm that learns which character sequences to merge into tokens |
| **Vocab size** | Total number of unique tokens the model knows. GPT-4 uses ~100K tokens |
| **Tokenizer** | The component that converts text → token IDs and back |

### Evaluation
| Term | What It Means |
|------|--------------|
| **val_bpb (validation bits per byte)** | How many bits the model needs to encode each byte of text. Lower = better. Used in autoresearch |
| **Perplexity** | How "surprised" the model is by the text. Lower = better. Related to loss |
| **Validation set** | Data the model never trains on, used to measure real performance |

### Optimization
| Term | What It Means |
|------|--------------|
| **Optimizer** | Algorithm that updates weights. Common: Adam, AdamW, SGD |
| **Muon** | A newer optimizer used in autoresearch that orthogonalizes gradients — works well for transformer matrices |
| **Weight decay** | Shrinks weights slightly each step to prevent overfitting |
| **Warmup** | Gradually increasing learning rate at the start of training for stability |
| **Mixed precision (bfloat16)** | Using 16-bit numbers instead of 32-bit for speed, with minimal accuracy loss |
| **torch.compile()** | PyTorch feature that JIT-compiles your model for faster execution |

### Infrastructure
| Term | What It Means |
|------|--------------|
| **GPU** | Graphics card repurposed for parallel math — essential for training |
| **VRAM** | GPU memory. Determines how large a model you can train |
| **CUDA** | NVIDIA's toolkit for GPU computing |
| **MPS** | Apple's GPU compute for M-series Macs |
| **Flash Attention** | Optimized attention implementation that uses less memory and is faster |

---

## Phase 4: Hands-On Projects (Week 4+)

### Project 1: Run autoresearch
Once you understand the fundamentals, run autoresearch yourself:
```bash
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch
uv sync
uv run prepare.py   # Download data, train tokenizer
uv run train.py     # Run a single 5-min training experiment
```
Then read `program.md` and watch the agent experiment autonomously.

### Project 2: nanoGPT
Karpathy's earlier project — train GPT-2 from scratch:
- **Repo**: https://github.com/karpathy/nanoGPT
- Simpler than autoresearch, great for learning

### Project 3: llm.c
LLM training in pure C/CUDA — no Python, no PyTorch:
- **Repo**: https://github.com/karpathy/llm.c
- For understanding what's happening "under the hood"

---

## Phase 5: Going Deeper

### Papers Worth Reading (after the course)
| Paper | Why |
|-------|-----|
| "Attention Is All You Need" (2017) | The original transformer paper |
| "Language Models are Few-Shot Learners" (GPT-3) | Scaling laws, in-context learning |
| "Training language models to follow instructions" (InstructGPT) | How RLHF works |
| "LLaMA: Open and Efficient Foundation Language Models" | How Meta built open-source LLMs |

### Communities
- **r/LocalLLaMA** (Reddit) — community of people running/fine-tuning open models
- **Hugging Face** — hub for models, datasets, and tutorials
- **karpathy/autoresearch GitHub Discussions** — active community experimenting

---

## Recommended Learning Order (TL;DR)

```
1. Watch "Intro to Large Language Models" (1 hour, conceptual overview)
2. Work through Zero to Hero playlist (code along!)
3. Watch "Deep Dive into LLMs like ChatGPT" (3.5 hours)
4. Run autoresearch on your machine
5. Read the autoresearch code (prepare.py, train.py) — you'll understand it now
6. Start your own project (Sinhala LLM, prompt optimization, etc.)
```

---

## Resources

- Karpathy's Zero to Hero: https://karpathy.ai/zero-to-hero.html
- nn-zero-to-hero GitHub: https://github.com/karpathy/nn-zero-to-hero
- Karpathy's website: https://karpathy.ai/
- microGPT blog post: http://karpathy.github.io/2026/02/12/microgpt/
- Dummy's Guide to autoresearch: https://x.com/hooeem/status/2030720614752039185
- Hugging Face learn: https://huggingface.co/learn
