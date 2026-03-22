# Prompt Optimization Using the Autoresearch Loop

How to use the Karpathy Loop pattern to autonomously find the best prompts for your agent tasks. Adapted for the Vault project — agents extracting information from transcripts.

---

## The Core Idea

Instead of the agent modifying `train.py` to improve `val_bpb`, the agent modifies **prompt files** to improve **extraction accuracy** (or whatever metric you define).

```
Original autoresearch:         Your adaptation:
─────────────────────         ─────────────────────
Artifact: train.py       →    Artifact: prompt.txt (or prompts/*.md)
Metric:   val_bpb         →    Metric:   extraction accuracy score
Budget:   5 min GPU       →    Budget:   N test cases evaluated
Loop:     modify code     →    Loop:     modify prompts
          train model                    run extraction
          measure loss                   score results
          keep/discard                   keep/discard
```

---

## How It Works for Prompt Optimization

### The 3 Components You Need

#### 1. The Artifact (what the agent modifies)
A prompt file — the system prompt, extraction instructions, few-shot examples, or output format specification that your agents use for transcript extraction.

```
prompts/
├── system_prompt.md        # Main system prompt (agent modifies this)
├── extraction_rules.md     # Rules for what to extract
└── few_shot_examples.md    # Example input/output pairs
```

#### 2. The Evaluation Harness (immutable — agent cannot touch)
A script that:
1. Runs your prompt against a fixed set of test transcripts
2. Compares extracted output against known-correct answers (ground truth)
3. Produces a single numeric score

```python
# eval_prompts.py — DO NOT MODIFY (this is your prepare.py equivalent)

import json
import sys

def load_test_cases():
    """Load transcript + expected extraction pairs."""
    with open("eval/test_cases.json") as f:
        return json.load(f)

def run_extraction(prompt_text, transcript):
    """Call your LLM with the prompt and transcript, return extracted data."""
    # Use your preferred API (Claude, Qwen, OpenAI, etc.)
    response = call_llm(
        system_prompt=prompt_text,
        user_message=transcript
    )
    return parse_response(response)

def score_extraction(extracted, expected):
    """Score a single extraction against ground truth.

    Returns a score between 0.0 and 1.0.
    You define what "correct" means for your use case.
    """
    score = 0.0
    total_fields = len(expected)

    for field, expected_value in expected.items():
        extracted_value = extracted.get(field, None)
        if extracted_value is None:
            continue
        if matches(extracted_value, expected_value):
            score += 1.0
        elif partial_match(extracted_value, expected_value):
            score += 0.5

    return score / total_fields if total_fields > 0 else 0.0

def evaluate():
    """Run full evaluation. Print final score."""
    # Load the current prompt (the artifact the agent modifies)
    with open("prompts/system_prompt.md") as f:
        prompt = f.read()

    test_cases = load_test_cases()
    scores = []

    for case in test_cases:
        extracted = run_extraction(prompt, case["transcript"])
        score = score_extraction(extracted, case["expected"])
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    # Output in parseable format (like autoresearch's val_bpb output)
    print(f"---")
    print(f"accuracy:      {avg_score:.6f}")
    print(f"total_cases:   {len(test_cases)}")
    print(f"perfect_cases: {sum(1 for s in scores if s == 1.0)}")
    print(f"zero_cases:    {sum(1 for s in scores if s == 0.0)}")

if __name__ == "__main__":
    evaluate()
```

#### 3. The Ground Truth Test Cases

```json
// eval/test_cases.json
[
  {
    "transcript": "So in Q3 we saw revenue hit 4.2 million, up 15% year over year. The main driver was our enterprise segment which grew 23%. We're projecting Q4 revenue of 4.8 million...",
    "expected": {
      "revenue": "4.2 million",
      "revenue_growth": "15% YoY",
      "segment": "enterprise",
      "segment_growth": "23%",
      "projection": "Q4 revenue of 4.8 million",
      "time_period": "Q3"
    }
  },
  {
    "transcript": "The patient presented with persistent headaches over 3 weeks. MRI showed no abnormalities. Prescribed ibuprofen 400mg twice daily. Follow-up in 2 weeks...",
    "expected": {
      "symptoms": "persistent headaches, 3 weeks duration",
      "findings": "MRI — no abnormalities",
      "treatment": "ibuprofen 400mg twice daily",
      "follow_up": "2 weeks"
    }
  }
]
```

> **Critical**: You need at least **20-30 test cases** for meaningful signal. More = better. These must cover edge cases, different transcript types, and tricky scenarios.

---

## Setting Up the Loop

### Directory Structure

```
vault-prompt-optimization/
├── program.md                  # Agent instructions (like autoresearch's program.md)
├── prompts/
│   └── system_prompt.md        # THE ARTIFACT — agent modifies this
├── eval/
│   ├── eval_prompts.py         # Evaluation harness — DO NOT MODIFY
│   ├── test_cases.json         # Ground truth — DO NOT MODIFY
│   └── run_eval.sh             # Wrapper script
├── results.tsv                 # Experiment log
└── .gitignore
```

### The run_eval.sh Script (Your autoresearch.sh Equivalent)

```bash
#!/usr/bin/env bash
# Run evaluation pipeline — tests prompts against ground truth
# Outputs METRIC lines for the agent to parse
set -euo pipefail

cd "$(dirname "$0")/.."

# Step 1: Validate prompt file exists and is non-empty
if [ ! -s prompts/system_prompt.md ]; then
  echo "FATAL: prompts/system_prompt.md is empty or missing"
  exit 1
fi

# Step 2: Run evaluation
echo "=== Running Prompt Evaluation ==="
EVAL_OUTPUT=$(python eval/eval_prompts.py 2>&1)
echo "$EVAL_OUTPUT"

# Step 3: Extract metrics
ACCURACY=$(echo "$EVAL_OUTPUT" | grep "^accuracy:" | awk '{print $2}')
PERFECT=$(echo "$EVAL_OUTPUT" | grep "^perfect_cases:" | awk '{print $2}')

echo ""
echo "METRIC accuracy=$ACCURACY"
echo "METRIC perfect_cases=$PERFECT"
```

### The program.md (Agent Instructions)

```markdown
# Vault Prompt Optimization

You are optimizing prompts for an information extraction system that processes
transcripts. Your goal: maximize extraction accuracy.

## Setup

1. Read all files for context:
   - This file (program.md)
   - prompts/system_prompt.md — THE FILE YOU MODIFY
   - eval/test_cases.json — understand what's being tested
   - eval/eval_prompts.py — understand how scoring works (DO NOT MODIFY)
2. Create branch: `git checkout -b autoexp/<tag>`
3. Initialize results.tsv with header row
4. Run baseline: `bash eval/run_eval.sh > run.log 2>&1`

## What You CAN Do
- Modify `prompts/system_prompt.md` — this is the only file you edit
- Change: wording, structure, instructions, few-shot examples, output format,
  chain-of-thought instructions, role definitions, constraints, etc.

## What You CANNOT Do
- Modify eval/eval_prompts.py or eval/test_cases.json
- Change the evaluation harness or test cases
- Add external tools or API calls

## The Metric
- **accuracy** (0.0 to 1.0) — higher is better
- **perfect_cases** — number of test cases with 100% extraction
- Goal: maximize accuracy. Higher = better prompt.

## Scoring Criteria
Every criterion is binary. The evaluation checks:
- Did the extraction find the correct value for each field?
- Partial matches get 0.5 credit
- Missing fields get 0 credit

## The Experiment Loop

LOOP FOREVER:

1. Review current prompt and recent results
2. Form a hypothesis about what might improve extraction
3. Modify prompts/system_prompt.md
4. git commit
5. Run: `bash eval/run_eval.sh > run.log 2>&1`
6. Read results: `grep "^METRIC" run.log`
7. If empty: crashed. Run `tail -50 run.log` to diagnose
8. Log to results.tsv (tab-separated):
   commit  accuracy  perfect_cases  status  description
9. If accuracy improved → keep the commit
10. If accuracy same or worse → git reset back

## Ideas to Try
- Add explicit extraction field definitions
- Add few-shot examples (input transcript → expected output)
- Try chain-of-thought ("First identify all entities, then...")
- Try structured output format (JSON schema)
- Constrain output format strictly
- Add error handling instructions ("If field is not mentioned, output null")
- Try role-based prompting ("You are an expert data analyst...")
- Simplify — sometimes shorter prompts work better
- Add negative examples ("Do NOT include...")
- Try step-by-step decomposition

## Simplicity Criterion
All else being equal, a shorter prompt that achieves the same accuracy is
better. A 0.01 improvement that doubles prompt length? Probably not worth it.
A 0.01 improvement from removing text? Definitely keep.

## NEVER STOP
Once the loop begins, do NOT pause to ask the human. Continue experimenting
until manually stopped.
```

---

## Running It

### With Claude Code
```bash
# In the vault-prompt-optimization directory
# Start Claude Code, then:
"Read program.md and let's kick off a new experiment. Do the setup first."
```

### With Any Agent
Point your agent at `program.md` and let it go. The loop is agent-agnostic — it works with Claude, Codex, Gemini, or any agent that can edit files and run shell commands.

---

## Designing Good Evaluation Criteria

This is the most important part. Bad criteria = useless optimization.

### Rules for Good Criteria

1. **Every criterion must be binary** — pass/fail, yes/no. NOT "how compelling is the extraction?" but "did the extraction include the revenue figure? yes/no"

2. **Don't change the eval mid-loop** — Lock test cases before starting. Changing scoring invalidates all previous results.

3. **Cover edge cases** — Include test cases where:
   - Information is implied, not stated directly
   - Numbers appear in different formats ("4.2M" vs "4,200,000")
   - Multiple values exist for one field
   - A field is genuinely absent from the transcript
   - The transcript is noisy (filler words, corrections, interruptions)

4. **Use enough test cases** — Minimum 20. Ideally 50+. With too few, improvements might be noise.

### Example Scoring Functions

```python
# Exact match (strict)
def matches(extracted, expected):
    return extracted.strip().lower() == expected.strip().lower()

# Fuzzy match (for text fields)
def partial_match(extracted, expected):
    # Check if key information is present
    expected_words = set(expected.lower().split())
    extracted_words = set(extracted.lower().split())
    overlap = len(expected_words & extracted_words) / len(expected_words)
    return overlap > 0.7

# Numeric match (for numbers)
def numeric_match(extracted, expected):
    try:
        e1 = parse_number(extracted)  # "4.2M" → 4200000
        e2 = parse_number(expected)
        return abs(e1 - e2) / max(abs(e2), 1) < 0.01  # within 1%
    except:
        return False
```

---

## What the Agent Typically Discovers

Based on real-world prompt optimization experiments:

1. **Shorter is often better** — Agents frequently improve accuracy by *removing* instructions that confuse the LLM
2. **Structured output format matters a lot** — JSON with explicit field names usually outperforms free-text
3. **Few-shot examples are high-value** — Even 2-3 examples can dramatically improve extraction
4. **Negative examples help** — "Do NOT include summaries, only extract explicit values" prevents hallucination
5. **Step-by-step decomposition** — "First read the full transcript, then identify each field one by one" often beats "Extract all fields"
6. **The final prompt is often surprising** — Agents find phrasings that humans wouldn't try

### Real-World Example: AutoVoiceEvals
In a documented case, 20 automated iterations improved a scheduling agent's success rate from **25% → 100%**. The final prompt was actually **shorter** than the starting prompt — the agent simplified while improving.

---

## Adapting for the Vault Project Specifically

For your use case (agents extracting info from transcripts), consider:

### Multiple Prompt Files
If your pipeline has multiple agents (e.g., one for entity extraction, one for summarization, one for action items), optimize each prompt separately:

```
prompts/
├── entity_extraction.md    # Optimize first (foundation)
├── summarization.md        # Optimize second
├── action_items.md         # Optimize third
└── final_synthesis.md      # Optimize last (depends on others)
```

### Cost Tracking
Add API cost tracking to your eval script:

```python
# Track tokens used per evaluation run
print(f"total_tokens:  {total_tokens}")
print(f"est_cost_usd:  {total_tokens * cost_per_token:.4f}")
```

### Multi-Metric Optimization
You might care about accuracy AND cost:

```
# In results.tsv
commit  accuracy  tokens_per_case  status  description
a1b2c3d 0.850     2400             keep    baseline
b2c3d4e 0.870     2200             keep    shorter prompt with examples
c3d4e5f 0.875     4800             discard added CoT but doubled tokens
```

---

## Reference: The Original Autoresearch Repo

The pattern comes from Karpathy's autoresearch:
- **Repo**: https://github.com/karpathy/autoresearch
- **How it works**: AI agent modifies `train.py`, runs 5-min training, measures `val_bpb`, keeps/discards
- **program.md**: The agent instructions that drive the loop

### Quick start (to see the original in action):
```bash
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch
uv sync
uv run prepare.py    # Download data, train tokenizer (~2 min)
uv run train.py      # Run a single 5-min training experiment
```

Then point your agent at `program.md`:
```
"Read program.md and let's kick off a new experiment. Do the setup first."
```

The agent will autonomously loop: modify → train → measure → keep/discard → repeat.

### Key files in the original repo:
| File | Purpose | Equivalent in your setup |
|------|---------|--------------------------|
| `train.py` | The artifact (agent modifies) | `prompts/system_prompt.md` |
| `prepare.py` | Fixed eval + data loading (read-only) | `eval/eval_prompts.py` |
| `program.md` | Agent instructions | `program.md` (same concept) |
| `results.tsv` | Experiment log | `results.tsv` (same format) |

### The generalized pattern (autoexp)
A community member generalized autoresearch into a domain-agnostic template:
- https://gist.github.com/adhishthite/16d8fd9076e85c033b75e187e8a6b94e

The 4 required components:
1. **Target file** — single file to modify
2. **Eval harness** — immutable script producing metrics
3. **Metric(s)** — 1-2 scalar values
4. **Budget** — time/cost/run limits

---

## Sources

- Karpathy's autoresearch: https://github.com/karpathy/autoresearch
- Autoexp (generalized loop): https://gist.github.com/adhishthite/16d8fd9076e85c033b75e187e8a6b94e
- AutoVoiceEvals (prompt optimization case study): Fortune article on the Karpathy Loop
- Shopify Liquid autoresearch.md: https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md
- MindStudio guide: https://www.mindstudio.ai/blog/autoresearch-optimize-business-metrics-autonomously
- Universal Skill adaptation: https://medium.com/@k.balu124/i-turned-andrej-karpathys-autoresearch-into-a-universal-skill-1cb3d44fc669
