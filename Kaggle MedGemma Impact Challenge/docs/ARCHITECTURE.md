# Architecture

## System Overview

CXR-Sentinel is a **directed acyclic graph (DAG) pipeline** that composes two Google HAI-DEF models into a unified diagnostic-to-treatment validation system. The architecture enforces a strict separation between generation and validation — the Diagnostician never evaluates its own work.

## Pipeline Flow

```
Input (CXR + Findings)
         │
         ▼
┌─────────────────────┐
│   DIAGNOSTICIAN      │  MedGemma 1.5 4B
│   Multimodal CoT     │  SigLIP encoder + Gemma 3
│   → JSON impression   │  384px CXR + text findings
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────────┐
│CHALLENGER│ │ FACT-CHECKER  │
│ 3 perturb│ │ Entity match  │
│  types   │ │ Synonym-aware │
└────┬─────┘ └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
      ┌─────────────┐
      │  HITL GATE   │
      │ HI/PSS/SCE   │
      └──────┬──────┘
        ┌────┴────┐
        ▼         ▼
   AUTO-PASS   REVIEW
                  │
                  ▼
            ┌──────────┐
            │ TxGemma   │
            │ 2B-Predict│
            │ ClinTox   │
            └──────────┘
```

## Agent Specifications

### Diagnostician (MedGemma 1.5 4B)

**Input:** CXR image (384px, SigLIP-encoded) + radiologist findings text  
**Process:** Severity-aware Chain-of-Thought prompting  
**Output:** Structured JSON with entities, severity, primary/secondary diagnoses

Cross-attends between visual features (opacity, cardiac silhouette, costophrenic blunting) and textual entities. Supports three modes: multimodal, image-only, text-only.

### Challenger (Adversarial)

Three perturbation types applied at inference time:

| Type | Method | What it tests |
|------|--------|---------------|
| Entity removal | Drop a random finding | Recall without explicit cues |
| Confounder injection | Add misleading context | Resistance to noise |
| Severity contradiction | Flip severity descriptors | Robustness to conflicting info |

**PSS** (Perturbation Stability Score) = fraction of core entities maintained across perturbations.

### FactChecker (Anti-Hallucination)

Synonym-aware entity extraction using a 20-entry clinical dictionary with canonical forms and variant mappings. Computes 7 metrics per case:

| Metric | Formula | Target |
|--------|---------|--------|
| DCR | \|G ∩ M\| / \|G\| | ≥ 0.70 |
| HI | \|M \ G\| / \|M\| | ≤ 0.10 |
| F1 | 2·P·R / (P+R) | ≥ 0.60 |
| ROUGE-L | LCS F-measure | ≥ 0.50 |
| PSS | Perturbation stability | ≥ 0.80 |
| CCS | Severity-weighted coverage | ≥ 0.70 |
| SCE | Severity calibration error | ≤ 0.25 |

### HITL Safety Gate

Conformal escalation based on composite risk. Triggers expert review when any threshold is exceeded:

- HI > 0.4 (high hallucination)
- PSS < 0.5 (unstable under perturbation)
- SCE > 0.5 (severity miscalibration)

**Design choice:** Conservative thresholds ensure 35% escalation rate. All high-severity-error cases were correctly routed to expert review in validation.

## Model Loading

MedGemma is loaded with 4-bit NF4 quantization via BitsAndBytes, requiring only 3 GB VRAM. This enables deployment on consumer GPUs (T4, RTX 3060+) and hospital workstations.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

## QLoRA Fine-Tuning

Rank-8 LoRA adapters fine-tuned on 60 real OpenI reports. Training target: structured JSON clinical extraction from radiologist findings.

- Loss: 2.41 → 2.20 (1 epoch)
- Time: ~15 min on T4
- VRAM: ~12 GB
- Adapters can be swapped for institutional adaptation without model re-download
