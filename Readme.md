# Agentic AI — Autonomous Bug Fixer & ML Pipeline

Fully autonomous pipeline: discovers GitHub repos with bugs → clones them → analyzes across **multiple files** → fixes bugs using your **local Ollama models** → runs and verifies the code → retries until it works → commits → opens a PR → explains every decision using **real SHAP/LIME/AST explainability** → optionally finds the right model on HuggingFace → prepares fine-tuning configs → publishes to HuggingFace. All local, no cloud AI needed.

---

## What it actually does

```
python main.py --topic "python ml bugs" --repos 2
```

That single command triggers this entire loop automatically:

```
1.  [PaperReader]   Optional: read arXiv paper → extract insights → enrich search topic
2.  [RepoHunter]    Search GitHub for SMALL Python repos (< 5MB, 10–5000 stars) with open issues
                    Skips massive repos like pytorch/tensorflow/cvat that local LLMs can't handle
3.  [RAGStore]      Save repo metadata to ChromaDB vector store for future semantic search
4.  [GitManager]    git clone --depth=1 into ./workspace/<repo-name>/
5.  [RepoHunter]    Fetch top 3 open bug-labelled issues from GitHub API

    ┌─ LOOP per issue ──────────────────────────────────────────────────┐
    │                                                                    │
    │  6.  [BugFixer]     Read ALL relevant files (up to 12), not just  │
    │                     one. Scores by keyword match, follows imports, │
    │                     includes requirements.txt for context.         │
    │                                                                    │
    │  7.  [OllamaClient] Send full multi-file context + issue to your  │
    │                     local model (DeepSeek-R1 / Qwen2.5-Coder /    │
    │                     Gemma2 etc). Timeout: 30 minutes per request. │
    │                     Returns complete fixed file contents.          │
    │                                                                    │
    │  8.  [BugFixer]     Cross-file impact check: scan ALL other files │
    │                     that import the changed modules. If any would  │
    │                     break, feed that warning back to the LLM and  │
    │                     retry with the constraint.                     │
    │                                                                    │
    │  9.  [XAIExplainer] Run real explainability BEFORE committing:    │
    │                     • AST diff: exact functions added/removed/     │
    │                       modified with cyclomatic complexity delta    │
    │                     • SHAP token importance: which words in the   │
    │                       issue text drove the fix decision            │
    │                     • Risk scoring: high/medium/low per change    │
    │                     • Cross-file export map: which public funcs   │
    │                       are now different and could affect callers  │
    │                     Writes AI_CHANGES.md into the fix branch.     │
    │                                                                    │
    │  10. [BugFixer]     Run: python <entry_file> inside cloned repo   │
    │                     Captures stdout + stderr. 45s run timeout.    │
    │                     Falls back to pytest if no entry file found.  │
    │                                                                    │
    │      If exit code 0 → go to step 11                               │
    │      If crash → inject stderr back into LLM prompt → retry        │
    │      (up to --attempts times, default 5)                          │
    │                                                                    │
    │  11. [GitManager]   git checkout -b fix/issue-N                   │
    │                     git add -A && git commit (AI_CHANGES.md incl) │
    │                     git push origin fix/issue-N                   │
    │                     Open PR with full XAI explanation in body     │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

12. [ModelManager]  Search HuggingFace for best GGUF model for this task type.
                    If you already have a good local model: explains why it's the
                    right choice and what quantization it's using.
                    If dataset found in repo: writes finetune_config.json with
                    step-by-step fine-tuning instructions + quantization guide.

13. [HFPublisher]   (only with --publish-hf) Upload fixed repo as HuggingFace Space.
                    Auto-generates README with explanation of what was fixed.
```

---

## XAI — Real Explainability (SHAP + LIME + AST)

Every fix gets a full explanation before it's committed. This is real XAI, not just LLM-generated text.

### AST-based code attribution
Uses Python's `ast` module to diff the syntax trees of original vs fixed files:
- Exact functions that were added, removed, or modified
- Cyclomatic complexity delta (did the fix make it more or less complex?)
- Risk scoring per change: HIGH (removed function), MEDIUM (modified function), LOW (new function / inline)
- Cross-file export map: which public functions changed and could affect callers

### SHAP token importance
Scores which words in the issue text had the highest correspondence with the fix:
- Maps issue vocabulary to fix code vocabulary
- Shows which problem keywords drove which code changes
- Approximates SHAP-style attribution on the prompt→fix relationship

### SHAP for ML models (when the repo contains a model)
If the repo has a trained sklearn/xgboost/lightgbm model:
- Runs `shap.TreeExplainer` (or `KernelExplainer` as fallback)
- Saves beeswarm plot (`shap_beeswarm.png`) — global feature importance
- Saves bar plot (`shap_importance.png`) — mean |SHAP| per feature
- Saves waterfall plot (`shap_waterfall.png`) — per-prediction explanation
- Writes `shap_explanation.json` with top features and plain-English interpretation

### LIME for text classification
If the repo classifies text:
- Runs `LimeTextExplainer` to show which words pushed the classification
- Saves `lime_explanation.html` (interactive in browser)
- Writes `lime_explanation.json` with per-word weights

### What gets committed
Every fix branch includes:
```
AI_CHANGES.md          ← Full XAI report (risk summary, AST analysis, plain-English explanation)
shap_explanation.json  ← If model found
shap_beeswarm.png      ← If model found
lime_explanation.json  ← If text classifier found
finetune_config.json   ← If training data found
```

---

## Setup

### Prerequisites
- **Docker Desktop** — for ChromaDB
- **Ollama** running on your laptop with at least one model:
  ```powershell
  ollama pull qwen2.5-coder:7b     # best for code fixes (recommended)
  ollama pull deepseek-r1:7b       # best for reasoning about bugs
  ollama pull smollm2:1.7b         # fast fallback for quick fixes
  ```
- **Python 3.11+** via uv

### Install
```powershell
# 1. Create env
uv venv .venv
.venv\Scripts\activate

# 2. Install dependencies (includes shap, lime, sklearn, matplotlib)
uv pip install -r requirements.txt

# 3. Copy and fill in tokens
copy .env.example .env
# Edit .env: add GITHUB_TOKEN and HF_TOKEN

# 4. Start ChromaDB
docker compose up -d chromadb

# 5. Health check
python main.py --check
```

### .env file
```env
GITHUB_TOKEN=ghp_...          # github.com/settings/tokens → repo + workflow scope
HF_TOKEN=hf_...               # huggingface.co/settings/tokens → write access
HF_USERNAME=your_username

OLLAMA_BASE_URL=http://localhost:11434
CHROMA_HOST=localhost
CHROMA_PORT=8000

MAX_REPOS=2
MAX_FIX_ATTEMPTS=5
AUTO_PR=true
AUTO_PUBLISH_HF=false
AUTO_FINETUNE=false           # Set true to auto-search HF models and prep fine-tuning
WORKSPACE_DIR=./workspace
```

---

## Running

```powershell
# Basic — find 2 small Python ML repos, fix bugs, open PRs
python main.py

# Custom topic
python main.py --topic "pytorch dataset loader bug" --repos 3

# More fix attempts (useful for complex bugs)
python main.py --attempts 10

# Read a research paper first, use insights to guide fixes
python main.py --paper https://arxiv.org/abs/2303.08774 --topic "LLM inference bugs"

# Publish fixed repos to HuggingFace after fixing
python main.py --publish-hf

# Watch live in a second terminal
python main.py --dashboard

# Health check
python main.py --check

# See your local models
python main.py --models

# Search HuggingFace for a model
python main.py --search-hf "deepseek coder quantized"

# Pull a new model into Ollama
python main.py --pull-model qwen2.5-coder:7b
```

---

## Why 30-minute timeouts?

Ollama needs time to:
1. Load the model weights into RAM/VRAM (can take 1–5 minutes cold start)
2. Process a large context window (12 files + issue + instructions = ~8000 tokens)
3. Generate a complete multi-file fix (JSON with full file contents = often 2000+ tokens output)

`deepseek-r1:7b` on CPU can take 10–20 minutes for a complex prompt. The timeout is set to 1800 seconds (30 minutes) for inference calls. If you have a GPU, it'll be much faster. Health checks and list calls still use short timeouts (5–10s).

---

## Repo selection logic

The pipeline avoids repos that are too large for a local LLM to analyze:

| Filter | Value | Why |
|---|---|---|
| Max size | 5 MB | Larger repos have too many files to fit in context |
| Max stars | 5,000 | Mega-repos (pytorch, cvat etc) need specialized domain knowledge |
| Min stars | 10 | Ensures real projects, not toy repos |
| Language | Python only | Local LLMs handle Python best |
| Has open issues | required | Nothing to fix otherwise |
| Blocklist | pytorch, tensorflow, cvat, keras, huggingface, numpy, scipy, pandas, django, flask... | Too complex |

---

## Multi-file fix logic

When the LLM generates a fix it's asked to:
1. Read ALL files in context (up to 12, scored by keyword relevance to the issue)
2. Fix EVERY file that contributes to the bug, not just the most obvious one
3. Ensure all call sites are updated if a function signature changes
4. Return a reason per file explaining why that specific file needed to change

After the fix is generated, the cross-file impact checker:
1. Finds all other files in the repo that import from any changed module
2. Asks the LLM whether those files would break
3. If yes: injects the warning back into the next attempt prompt
4. Loop continues until the fix is safe across the whole codebase

---

## HuggingFace model pipeline

When `AUTO_FINETUNE=true` in `.env`, after fixing a repo the pipeline:

1. Checks your local Ollama models and picks the best one for the task type
2. If no good local model: searches HuggingFace for top GGUF downloads
3. Explains the choice: why this model, what quantization, tradeoffs
4. If a dataset is found in the repo (`.jsonl`, `.csv`, `data/*.json`): writes `finetune_config.json` with:
   - Recommended base model
   - Quantization choice with explanation (Q4_K_M vs Q8_0 vs F16)
   - Step-by-step fine-tuning instructions using unsloth or llama.cpp
   - Export to GGUF + push to HuggingFace steps

### Quantization guide
| Format | RAM (7B model) | Quality loss | When to use |
|---|---|---|---|
| Q4_K_M | ~4 GB | Small | Default — best balance |
| Q5_K_M | ~5 GB | Very small | More VRAM available |
| Q8_0 | ~8 GB | Minimal | Quality matters most |
| Q2_K | ~2.5 GB | Noticeable | Very limited RAM |
| F16 | ~14 GB | None | Fine-tuning only |

---

## Project structure

```
Ml-agent/
├── main.py                    ← CLI entry point (all commands)
├── .env                       ← Your secrets (never commit)
├── .env.example               ← Template
├── requirements.txt           ← All deps including shap, lime
├── Dockerfile
├── docker-compose.yml
│
├── core/
│   └── orchestrator.py        ← Master coordinator — runs the full loop
│
├── agents/
│   ├── repo_hunter.py         ← GitHub search with size/complexity filters
│   ├── bug_fixer.py           ← Multi-file analysis + cross-file impact check
│   ├── xai_explainer.py       ← SHAP + LIME + AST attribution
│   ├── model_manager.py       ← HF model search + quantization + fine-tune prep
│   ├── hf_publisher.py        ← Publish to HuggingFace Spaces/Models
│   └── paper_reader.py        ← arXiv/PDF reader via MinerU or pdfminer
│
├── tools/
│   ├── ollama_client.py       ← Async Ollama wrapper (1800s timeout)
│   ├── git_manager.py         ← Clone / commit / push / open PR
│   └── rag_store.py           ← ChromaDB memory (repos, fixes, papers)
│
├── config/
│   └── settings.py            ← All config via .env
│
├── ui/
│   └── dashboard.py           ← Rich TUI live dashboard
│
├── workspace/                 ← Cloned repos live here
└── logs/
    ├── orchestrator.log       ← Full UTF-8 log
    └── session.jsonl          ← Structured event stream (dashboard reads this)
```

---

## What gets written into each fix branch

```
repo/
├── <original files — fixed>
├── AI_CHANGES.md              ← Full XAI report
│   ├── Summary (risk level, what changed)
│   ├── Root cause explanation
│   ├── Per-file: what changed, why, risk score
│   ├── AST analysis: functions added/removed/modified
│   ├── Token importance: which issue words drove the fix
│   └── Cross-file safety assessment
├── shap_explanation.json      ← If ML model found in repo
├── shap_beeswarm.png          ← SHAP global feature importance plot
├── shap_waterfall.png         ← SHAP per-prediction plot
├── lime_explanation.json      ← If text classifier found
├── lime_explanation.html      ← Interactive LIME report
└── finetune_config.json       ← If training data found
```

---

## Troubleshooting

**Ollama timing out:**
The default inference timeout is 1800s (30 minutes). If it still times out, your model may be too large for your hardware. Try a smaller model:
```powershell
ollama pull smollm2:1.7b    # Fastest, fits in ~2GB RAM
ollama pull qwen2:1.5b      # Small but capable
```

**Still picking huge repos:**
The size filter is `< 5MB`. If a repo passes but is still too complex, add its org to the `BLOCKLIST` in `agents/repo_hunter.py`.

**ChromaDB connection error:**
```powershell
docker compose up -d chromadb
# Wait 10s then retry
curl http://localhost:8000/api/v2/heartbeat
```

**Ollama not found:**
```powershell
# Make sure Ollama is running (leave this terminal open)
ollama serve
```

**SHAP install fails:**
```powershell
uv pip install shap --no-build-isolation
# or
pip install shap scikit-learn matplotlib
```

**Unicode errors on Windows:**
Already fixed — the orchestrator patches stdout/stderr to UTF-8 on startup. If you still see them, run:
```powershell
$env:PYTHONIOENCODING = "utf-8"
python main.py
```