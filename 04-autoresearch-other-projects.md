# Autoresearch Applied: Projects Beyond LLM Training

A catalog of how people have applied the Karpathy Loop pattern (modify → test → keep/discard → repeat) to domains beyond neural network training.

---

## The Pattern

Every application follows the same structure:

```
┌─────────────────────────────────────────────┐
│  1. Define an ARTIFACT the agent can modify  │
│  2. Define a METRIC to score it              │
│  3. Define an EVAL that runs in minutes      │
│  4. LOOP: modify → eval → keep/discard       │
└─────────────────────────────────────────────┘
```

---

## 1. Software Performance Optimization

### Shopify Liquid Template Engine

**Who**: Tobi Lütke (CEO of Shopify)
**What**: Optimized Liquid, the 20-year-old templating engine that renders every Shopify storefront
**Artifact**: Ruby source code in `lib/liquid/*.rb`
**Metric**: Combined parse+render time (microseconds) + object allocations
**Eval**: Unit tests → liquid-spec conformance → performance benchmark

**Results**:
- 120 automated experiments, 93 commits kept
- **53% faster parse+render**
- **61% fewer memory allocations**
- Parse time: 7,374 µs → ~3,400 µs

**How it was set up**:
- `auto/autoresearch.md` — agent instructions defining the objective, benchmark methodology, scope, and success criteria
- `auto/autoresearch.sh` — shell script that runs unit tests, spec conformance, then benchmarks. Only proceeds to benchmark if tests pass
- Agent could only modify files in `lib/liquid/` — tests and benchmarks were read-only
- Each change validated against full test suite before benchmarking

**Specific optimizations the agent found**:
- Replaced StringScanner tokenizer with `String#byteindex` (single-byte search ~40% faster than regex)
- Fast-path strategies for common template patterns
- Memory allocation reductions through object reuse

**Links**:
- PR: https://github.com/Shopify/liquid/pull/2056
- autoresearch.md: https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md
- autoresearch.sh: https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.sh
- Simon Willison writeup: https://simonwillison.net/2026/Mar/13/liquid/

---

## 2. Prompt Engineering / Voice Agents

### AutoVoiceEvals — Scheduling Agent Optimization

**Who**: Reported in Fortune article on the Karpathy Loop
**What**: Optimized prompts for an AI scheduling agent
**Artifact**: System prompt for a voice-based scheduling bot
**Metric**: Successful scheduling completion rate (%)
**Eval**: Run scheduling scenarios against the agent, score pass/fail

**Results**:
- 20 automated iterations
- Success rate: **25% → 100%**
- Final prompt was **shorter** than the original
- Agent discovered that removing instructions actually improved performance

**Key insight**: The agent simplified while improving — proving that human-written prompts often contain contradictory or confusing instructions that hurt performance.

---

## 3. Language Model Optimization (Smaller Models)

### Shopify QMD Query-Expansion Model

**Who**: Tobi Lütke
**What**: Optimized a query-expansion model for Shopify's search
**Artifact**: Model training configuration
**Metric**: Query expansion quality score

**Results**:
- 37 experiments in 8 hours
- A **0.8B parameter model scored 19% higher** than the previous hand-tuned 1.6B model
- Smaller model outperforming larger model = cost savings in production

---

## 4. Scaled Parallel Research

### SkyPilot — GPU Cluster Autoresearch

**Who**: SkyPilot team
**What**: Gave autoresearch access to 16 GPUs instead of 1
**Artifact**: train.py (same as original autoresearch)
**Metric**: val_bpb
**Eval**: Standard autoresearch training loop

**Results**:
- ~910 experiments in 8 hours
- val_bpb: 1.003 → 0.974 (2.87% improvement)
- Reached same best loss **9x faster** than sequential (8 hours vs ~72 hours)

**What the agent did differently with more GPUs**:
- Ran **factorial grids** (10-13 experiments per wave) catching interaction effects between parameters that sequential search misses
- Tested 6 model widths in a single wave, identified the trend immediately
- Taught itself to use H200s for validation while screening ideas on H100s
- Biggest late-stage find: `muon_beta2=0.98` (up from 0.95) — smoothed gradient normalization, worth ~0.001 val_bpb alone

**Link**: https://blog.skypilot.co/scaling-autoresearch/

---

## 5. Distributed Agent Network

### Hyperspace Network — 35 Agents Overnight

**Who**: Hyperspace network
**What**: 35 autonomous agents ran autoresearch simultaneously
**Artifact**: train.py variants
**Metric**: val_bpb

**Results**:
- 333 experiments completed in one night, completely unsupervised
- Demonstrated the SETI@home-style distributed research vision Karpathy described

---

## 6. Consumer Hardware Experiments

### Mac Mini M4 — Overnight Research on Consumer Hardware

**Who**: Anonymous user documented on X
**What**: Ran autoresearch overnight on a Mac Mini M4 (not a datacenter GPU)
**Artifact**: train.py (adapted for MPS/CPU)
**Metric**: val_bpb

**Results**:
- 35 experiments attempted
- 26 failed or crashed (consumer hardware limitations)
- **7 succeeded** and revealed the model **"got better by getting simpler"**
- Agent removed complexity while improving the metric

**Key insight**: Even on consumer hardware with high failure rates, the loop still produces valuable discoveries.

---

## 7. Marketing & Business Metrics

### The Marketing Experiment Loop

**Who**: Eric Siu (founder of Single Grain ad agency)
**What**: Applied the autoresearch pattern to marketing optimization
**Artifact**: Ad copy, landing pages, email subject lines
**Metric**: Conversion rate, click-through rate, CAC
**Eval**: A/B test results

**Vision**:
- "Most marketing teams run ~30 experiments a year. The next generation will run 36,500+"
- Each experiment: modify copy → run A/B test → measure → keep/discard
- Agents run experiments while the team sleeps

### Greg Isenberg — Landing Page Optimization

**Who**: Greg Isenberg
**What**: Proposed applying the loop to:
- Landing page conversion rate
- Customer acquisition cost (CAC)
- Ad headline performance
**Insight**: "Give it a goal like 'find a higher converting landing page' or 'lower customer acquisition cost' — then it runs experiments 24/7"

---

## 8. SEO / AI Search Optimization

### AEO Agent Armies

**Who**: Various SEO practitioners
**What**: Applied the autoresearch pattern to content optimization for AI-powered search (ChatGPT, Perplexity, Google AI Overviews)
**Artifact**: Content pages, meta descriptions, structured data
**Metric**: AI search visibility, click-through from AI answers
**Eval**: Search ranking position, traffic from AI sources

**Reported results**: 920% average lift in AI-driven traffic (though this claim should be treated with skepticism)

---

## 9. Generalized Framework

### Autoexp — Domain-Agnostic Autoresearch

**Who**: Adhish Thite (community member)
**What**: Generalized autoresearch into a template for any domain
**Link**: https://gist.github.com/adhishthite/16d8fd9076e85c033b75e187e8a6b94e

**The 4 components**:
1. **Target file** — single file to modify (e.g., `train.py`, `prompt.txt`, `config.yaml`)
2. **Eval harness** — immutable script producing metrics
3. **Metric(s)** — 1-2 scalar values (↑ or ↓)
4. **Budget** — time/cost/run limits

**Applicable domains listed**:
- ML training, prompt engineering, RAG configs
- Performance tuning, API optimization
- System prompts, infrastructure-as-code

### Universal Claude Skill

**Who**: Udit Goenka
**What**: Converted autoresearch into a reusable Claude Skill for any optimization task
**Link**: https://x.com/iuditg/status/2032478521256509896
- Open source under MIT license
- Adapted for marketing, sales, research, optimization

### pi-autoresearch

**Who**: davebcn87
**What**: Autonomous experiment loop extension for the pi framework
**Link**: https://github.com/davebcn87/pi-autoresearch

---

## 10. Academic Research

### AutoResearchClaw — Full Paper Generation

**Who**: Huaxiu Yao (ML researcher)
**What**: Extended the autoresearch concept to generate complete conference papers
**Artifact**: Research code + paper draft
**Metric**: Paper quality scores (automated evaluation)
**Eval**: Run experiments, generate results, write paper sections

**Claim**: "One message in, full conference paper out. Real experiments. Real citations. Real code. No human in the loop."

**Link**: https://x.com/HuaxiuYaoML/status/2033038170653405308

---

## 11. Small Language Model Adoption

### Philipp Schmid — How Autoresearch Changes SLMs

**Who**: Philipp Schmid (Hugging Face)
**What**: Analysis of how autoresearch enables practical adoption of Small Language Models
**Insight**: The loop is particularly powerful for SLMs because:
- Training time is minutes, not hours
- Experiments are cheap
- The agent can rapidly explore architecture/hyperparameter space
- Optimal SLM for your specific use case found overnight

**Link**: https://www.philschmid.de/autoresearch

---

## Summary Table

| Project | Domain | Artifact | Metric | Notable Result |
|---------|--------|----------|--------|----------------|
| Shopify Liquid | Software perf | Ruby source | Parse+render time | 53% faster, 61% fewer allocations |
| AutoVoiceEvals | Prompt eng | System prompt | Success rate | 25% → 100%, shorter prompt |
| QMD Model | ML/Search | Training config | Quality score | 0.8B beat 1.6B by 19% |
| SkyPilot | Parallel ML | train.py | val_bpb | 9x faster with 16 GPUs |
| Hyperspace | Distributed ML | train.py | val_bpb | 333 experiments, 35 agents |
| Mac Mini M4 | Consumer ML | train.py | val_bpb | Simpler = better |
| Marketing Loop | Marketing | Ad copy | Conversion rate | 36,500 experiments/year vision |
| AEO Agents | SEO | Content | AI search visibility | 920% traffic lift (claimed) |
| AutoResearchClaw | Academic | Code + paper | Paper quality | Full papers generated |
| Autoexp | Any | Any file | Any metric | Generalized framework |

---

## What HASN'T Been Done (Opportunities)

As of March 2026, these applications haven't been publicly documented but are natural fits:

| Domain | Artifact | Metric | Notes |
|--------|----------|--------|-------|
| Classical ML (sklearn) | Model pipeline code | Accuracy/F1 | XGBoost, Random Forest hyperparameter search |
| Database query optimization | SQL queries | Query time | Optimize slow queries overnight |
| CSS/UI optimization | Stylesheet | Lighthouse score | Performance + accessibility |
| API rate limiting | Config files | Throughput + error rate | Find optimal limits |
| Infrastructure as Code | Terraform/k8s configs | Cost + latency | Optimize cloud spend |
| Game AI | Agent behavior code | Win rate | Autonomous game agent improvement |
| Compiler flags | Build config | Binary size + speed | Find optimal compilation settings |
| RAG pipeline | Chunking + retrieval code | Retrieval accuracy | Optimize chunk size, overlap, embedding model |

---

## How to Start Your Own

1. **Pick a domain** where you have a measurable metric
2. **Create the eval first** — this is the hardest and most important part
3. **Write your program.md** — adapt from autoresearch's or Shopify's
4. **Start simple** — one file, one metric, one agent
5. **Let it run overnight** — most gains come after cycle 5-6

The pattern works anywhere you can answer: "Is this version better than the last one?" with a number.

---

## Sources

- Karpathy's autoresearch: https://github.com/karpathy/autoresearch
- Fortune: The Karpathy Loop: https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/
- VentureBeat: https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai
- SkyPilot scaling: https://blog.skypilot.co/scaling-autoresearch/
- Shopify Liquid PR: https://github.com/Shopify/liquid/pull/2056
- Simon Willison on Liquid: https://simonwillison.net/2026/Mar/13/liquid/
- The New Stack: https://thenewstack.io/karpathy-autonomous-experiment-loop/
- Philipp Schmid on SLMs: https://www.philschmid.de/autoresearch
- MindStudio guide: https://www.mindstudio.ai/blog/autoresearch-optimize-business-metrics-autonomously
- Autoexp gist: https://gist.github.com/adhishthite/16d8fd9076e85c033b75e187e8a6b94e
- Karpathy on X (results): https://x.com/karpathy/status/2031135152349524125
- Karpathy on X (vision): https://x.com/karpathy/status/2030705271627284816
- Greg Isenberg on X: https://x.com/gregisenberg/status/2031870162702381515
- MarkTechPost: https://www.marktechpost.com/2026/03/08/andrej-karpathy-open-sources-autoresearch-a-630-line-python-tool-letting-ai-agents-run-autonomous-ml-experiments-on-single-gpus/
