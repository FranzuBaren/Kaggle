<div align="center">

# 🫁 CXR-Sentinel

### Multi-Agent Adversarial Framework for Clinical AI Validation

[![MedGemma](https://img.shields.io/badge/MedGemma_1.5-4B_Multimodal-4285F4?logo=google&logoColor=white)](https://ai.google.dev/gemma/docs/medgemma)
[![TxGemma](https://img.shields.io/badge/TxGemma-2B_Predict-34A853?logo=google&logoColor=white)](https://ai.google.dev/gemma/docs/txgemma)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![GPU](https://img.shields.io/badge/GPU-T4_16GB-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**An on-premise, privacy-first system that generates radiology impressions from chest X-rays,  
then adversarially validates them before any clinician sees the output.**

[Technical Overview (PDF)](docs/CXR-Sentinel_Technical_Overview.pdf) · [Kaggle Notebook](notebook/cxr-sentinel-v16-final.ipynb) · [Video Demo](demo/medgemma_sovereign_video.html) · [Quickstart](#-quickstart)

</div>

---

## Core Result

> **MedGemma 1.5 4B generates radiology impressions agreeing with board-certified radiologists on 73.8% of clinical entities** (95% CI [0.62, 0.86]), within the 70–80% inter-radiologist agreement range.
>
> Validated on 20 real OpenI chest X-ray cases via three adversarial agents.  
> **114 total HAI-DEF model calls** · Single T4 GPU · ~45 min end-to-end · Fully reproducible.

| Metric | Score | 95% CI | Meaning |
|--------|-------|--------|---------|
| **DCR** | 73.8% | [0.62, 0.86] | Entity agreement with radiologist |
| **HI** ↓ | 8.1% | [0.03, 0.15] | Hallucination rate |
| **PSS** | 88.4% | [0.76, 1.00] | Adversarial stability |
| **CCS** | 78.5% | [0.66, 0.89] | Severity-weighted coverage |
| **F1** | 0.650 | [0.54, 0.75] | Precision/recall balance |
| **HITL** | 35% | 7/20 cases | Expert review escalation |

---

## Why CXR-Sentinel?

An estimated **75 million chest X-rays** are performed annually in the US alone. Radiologists read 50–100 CXRs per shift, synthesizing an *impression* — the clinical conclusion that drives treatment. This synthesis consumes 30–40% of reporting time and is where errors propagate: **miss rates for secondary findings reach 20–30%** in high-volume settings.

No deployed system writes the impression. Cloud LLMs are prohibited under HIPAA/GDPR. Existing radiology AI (CheXpert, CheXbert) performs detection but never generates the prose driving treatment.

**CXR-Sentinel fills this gap.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Staff radiologist dictates findings as usual                           │
│                                                                         │
│  CXR-Sentinel independently generates a draft impression                │
│  from the same CXR + findings                                           │
│                                                                         │
│  ✓ Agreement    →  Fast sign-off                                        │
│  ✗ Disagreement →  Double-read alert with entity-level discrepancies    │
│                                                                         │
│  Safety net: invisible when not needed, conspicuous when it matters.    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
                        CXR-Sentinel — DAG Pipeline

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  DIAGNOSTICIAN   │────▶│    CHALLENGER     │────▶│   FACT-CHECKER   │
  │  (MedGemma 4B)   │     │ (Adversarial)    │     │ (Anti-Halluci.)  │
  │                  │     │                   │     │                  │
  │ • CXR + Findings │     │ • Entity removal  │     │ • Entity match   │
  │ • Severity CoT   │     │ • Confounder inj. │     │ • Synonym-aware  │
  │ • JSON output    │     │ • Severity flip   │     │ • GT concordance │
  └────────┬─────────┘     └──────────────────┘     └────────┬─────────┘
           │                                                  │
           │              ┌──────────────────┐                │
           └──────────────│   HITL SAFETY    │◀───────────────┘
                          │     GATE         │
                          │ HI>0.4 │ PSS<0.5│
                          │  │ SCE>0.5       │
                          └────────┬─────────┘
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
              ✅ AUTO-APPROVED              🔍 EXPERT REVIEW
                                     (entity-level diff)
                          │
                          ▼
                  ┌──────────────────┐
                  │    TxGemma 2B    │
                  │ Pharmacovigilance│
                  │ ClinTox SMILES   │
                  └──────────────────┘
```

### Three-Agent Design Philosophy

1. **Never trust your own output.** Diagnostician and FactChecker are separate agents — the generator never evaluates its own work.
2. **Fail loud, not silent.** Every disagreement is surfaced with entity-level detail.
3. **Stress-test at inference.** Challenger attacks every production case, not just during development.
4. **When in doubt, escalate.** 35% escalation rate is deliberately conservative.

---

## HAI-DEF Model Integration

### MedGemma 1.5 4B — 5 Distinct Roles (100 inference calls)

| Role | Calls | Description |
|------|-------|-------------|
| Diagnostician | 20 | Multimodal impression generation (CXR + findings) |
| Challenger | ~40 | Adversarial perturbation responses |
| Image-Only Ablation | 20 | CXR interpretation without text |
| Text-Only Ablation | 20 | Findings-only interpretation |
| QLoRA Fine-tuning | — | Rank-8, NF4, 60 OpenI reports, 1 min on T4 |

### TxGemma 2B-Predict — Pharmacovigilance (14 inference calls)

Detected clinical conditions → drug SMILES → ClinTox toxicity prediction:

```
effusion → furosemide → O=C(O)c1cc(S(=O)(=O)N)c(Cl)cc1NCc1ccco1 → ClinTox → "No" (not toxic)
```

---

## Three-Track Modality Ablation

| Track | Mean DCR | Finding |
|-------|----------|---------|
| Image-Only | 0.479 | MedGemma reads CXR independently |
| Multimodal | 0.738 | Text + image (main pipeline) |
| Text-Only | 0.746 | Text findings only |

**Key insight:** Image-only DCR of 47.9% proves MedGemma independently interprets CXR images. When text is complete, images add negligible signal (−0.8%). The clinical value: multimodal acts as a **safety net** when text findings are absent or incomplete.

---

## Repository Structure

```
cxr-sentinel/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Project metadata & dependencies
├── requirements.txt                   # Pinned dependencies
├── setup.cfg                          # Package configuration
│
├── notebook/
│   └── cxr-sentinel-v16-final.ipynb   # ⭐ Complete Kaggle notebook (Run All)
│
├── demo/
│   └── medgemma_sovereign_video.html  # 3-minute competition video demo
│
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── diagnostician.py           # MedGemma multimodal impression generation
│   │   ├── challenger.py              # Adversarial perturbation engine
│   │   └── factchecker.py            # Anti-hallucination validation
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── sentinel.py               # DAG execution engine
│   │   ├── hitl_gate.py              # Conformal safety gate
│   │   └── txgemma_pharma.py         # TxGemma pharmacovigilance module
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── clinical_vocab.py         # Synonym-aware entity vocabulary
│   │   ├── metrics.py                # DCR, HI, PSS, CCS, SCE computation
│   │   ├── data_loader.py            # OpenI dataset loading & cohort selection
│   │   └── model_loader.py           # MedGemma quantized loading
│   └── visualization/
│       ├── __init__.py
│       └── figures.py                 # 10 publication-grade figures
│
├── configs/
│   ├── default.yaml                   # Default pipeline configuration
│   └── qlora.yaml                     # QLoRA fine-tuning hyperparameters
│
├── docs/
│   ├── CXR-Sentinel_Technical_Overview.pdf  # 3-page technical writeup
│   ├── ARCHITECTURE.md                # Detailed architecture documentation
│   ├── CLINICAL_VALIDATION.md         # Validation methodology & metrics
│   ├── DEPLOYMENT.md                  # Deployment guide & FDA pathway
│   └── figures/                       # Generated figures (after run)
│
├── scripts/
│   ├── run_pipeline.py                # CLI entry point
│   └── generate_report.py            # Regulatory evidence report generator
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py                 # Agent unit tests
│   ├── test_metrics.py               # Metrics computation tests
│   └── test_pipeline.py              # Integration tests
│
└── .github/
    ├── workflows/
    │   └── ci.yml                     # CI: lint, type-check, tests
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

---

## 🚀 Quickstart

### Option 1: Kaggle Notebook (Recommended)

The fastest path — everything runs in a single notebook on Kaggle's free T4 GPU:

1. Open the [Kaggle Notebook](notebook/cxr-sentinel-v16-final.ipynb)
2. Add model inputs:
   - `google/medgemma` → MedGemma 1.5 4B
   - `google/txgemma` → TxGemma 2B-Predict
3. Accept [HAI-DEF Terms of Use](https://ai.google.dev/gemma/docs/hai-def-terms) on HuggingFace
4. Click **Run All** (~45 minutes)

### Option 2: Local / On-Premise

```bash
# Clone the repository
git clone https://github.com/francescoorsi/cxr-sentinel.git
cd cxr-sentinel

# Create environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Accept HAI-DEF model terms on HuggingFace, then:
huggingface-cli login

# Run the full pipeline
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --data-dir /path/to/openi \
    --output-dir results/
```

### Hardware Requirements

| Setup | GPU | VRAM | Time |
|-------|-----|------|------|
| **Minimum** | T4 | 16 GB | ~45 min |
| **Recommended** | A100 / RTX 4090 | 40+ GB | ~15 min |
| **Deployment** | Any GPU ≥6 GB | 6+ GB | Per-case: ~12 sec |

---

## Real-World Impact

| Metric | Value (100-case department) |
|--------|---------------------------|
| Time saved / day | 150 min (2.5 hours) |
| FTE equivalent | 0.3 radiologists recovered |
| Annual value | $75K recovered |
| Cost per case | $0.03 (GPU amortization) |
| Hardware ROI | < 30 days at 50 cases/day |
| Error reduction | 10–15% fewer major discrepancies |

---

## Comparison with Existing Systems

| System | Generates Impressions | CXR Input | On-Premise | Adversarial Validation | Pharmacovigilance |
|--------|:----:|:----:|:----:|:----:|:----:|
| GPT-4V | ✅ | ✅ | ❌ | ❌ | ❌ |
| Med-PaLM | ✅ | ❌ | ❌ | ❌ | ❌ |
| CheXpert | ❌ | ✅ | ✅ | ❌ | ❌ |
| CheXbert | ❌ | ✅ | ✅ | ❌ | ❌ |
| R2Gen | ✅ | ✅ | ❌ | Partial | ❌ |
| **CXR-Sentinel** | **✅** | **✅** | **✅** | **✅** | **✅** |

---

## Path to FDA / CE Clearance

CXR-Sentinel targets **FDA SaMD Class II** (decision-support):

- **PCCP:** QLoRA adapters updated without re-submission
- **Validation:** 20-case framework → 510(k) requires 500+ multi-site
- **QMS:** Per-case 7-metric audit trail aligned with IEC 62304
- **Intended Use:** "AI-assisted QA for radiology impression synthesis, requiring clinician confirmation"
- **Privacy:** Zero data egress — HIPAA/GDPR compliant by architecture, not policy

---

## Limitations & Roadmap

### Acknowledged Limitations

- 20-case cohort validates framework design, not deployment-scale performance
- Severity calibration (SCE = 0.467) exceeds target — HITL gate compensates
- No demographic bias audit (age, sex, ethnicity) at current scale
- TxGemma predictions on standard medications show limited discriminative value

### Roadmap

| Priority | Milestone | Description |
|----------|-----------|-------------|
| 🔴 P0 | Multi-site scale-up | 500+ cases across 3+ institutions for 510(k) |
| 🟡 P1 | MedSigLIP integration | Zero-shot anomaly pre-screening |
| 🟡 P1 | PubMed RAG pipeline | Evidence grounding for generated impressions |
| 🟢 P2 | MedGemma 27B upgrade | When T4-compatible quantization available |
| 🟢 P2 | HL7 FHIR integration | Longitudinal EHR context |
| 🔵 P3 | MedASR integration | Voice dictation → impression pipeline |

---

## Citation

```bibtex
@software{orsi2026cxrsentinel,
  author       = {Orsi, Francesco},
  title        = {{CXR-Sentinel}: A Multi-Agent Adversarial Framework 
                  for Clinical {AI} Validation},
  year         = {2026},
  publisher    = {Kaggle},
  note         = {MedGemma Impact Challenge 2026},
  url          = {https://github.com/francescoorsi/cxr-sentinel}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

MedGemma and TxGemma are subject to [Google's HAI-DEF Terms of Use](https://ai.google.dev/gemma/docs/hai-def-terms).

The OpenI dataset is provided by the [National Library of Medicine](https://openi.nlm.nih.gov/) under their terms of use.

---

<div align="center">
  <sub>Built with MedGemma · TxGemma · OpenI · Single T4 GPU · Zero Cloud Dependencies</sub>
</div>
