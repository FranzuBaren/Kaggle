# Deployment Guide

## Privacy Architecture

CXR-Sentinel is designed for **zero data egress** — no patient data touches any network or cloud API. This is HIPAA/GDPR compliance by architecture, not by policy.

```
┌──────────────────────────────────────────┐
│        Hospital Network (Air-gapped)      │
│                                            │
│  PACS ──► CXR-Sentinel ──► Report System  │
│                                            │
│  ✓ All inference on-premise               │
│  ✓ No internet at runtime                 │
│  ✗ No cloud API calls                     │
│  ✗ No patient data leaves the network     │
└──────────────────────────────────────────┘
```

## Hardware Requirements

| Tier | GPU | VRAM | Cases/Day | Cost |
|------|-----|------|-----------|------|
| Minimum | T4 / RTX 3060 | 6+ GB | ~50 | ~$800 |
| Standard | RTX 4090 | 24 GB | ~200 | ~$2,000 |
| Enterprise | A100 | 40+ GB | ~500+ | ~$10,000 |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- 50 GB disk space (models + data)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/francescoorsi/cxr-sentinel.git
cd cxr-sentinel

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Accept HAI-DEF terms and download models
huggingface-cli login
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('google/medgemma-4b-it')"

# 5. Download OpenI dataset
# From: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

# 6. Run pipeline
python scripts/run_pipeline.py --data-dir data/openi --output-dir results/
```

## Integration Points

### DICOM / PACS
- Input: DICOM images via pydicom or PACS query
- CXR images converted to 384px PNG for SigLIP encoder
- DICOM tag scrubbing for de-identification in production

### HL7 FHIR
- Output: Structured impression as FHIR DiagnosticReport resource
- Integration with hospital EHR for longitudinal context

### Report Templates
- Auto-drafts impressions into PACS-compatible report templates
- Radiologist reviews and signs off or modifies

## Institutional Adaptation

QLoRA adapters enable rapid institutional customization:

1. Collect 60+ local radiology reports
2. Run QLoRA fine-tuning (~15 min on T4)
3. Swap adapter without re-downloading base model
4. Validate on local held-out set

## FDA SaMD Class II Pathway

CXR-Sentinel targets FDA Software as a Medical Device (SaMD) Class II clearance via 510(k):

| Requirement | Current Status | For 510(k) |
|-------------|---------------|-------------|
| Validation cohort | 20 cases | 500+ multi-site |
| Intended use statement | ✅ Defined | ✅ |
| Audit trail | ✅ 7-metric per case | ✅ IEC 62304 aligned |
| PCCP | ✅ QLoRA adapter updates | ✅ No re-submission |
| Demographic audit | ❌ Not yet | Required |
| QMS documentation | Partial | Full IEC 62304 |

**Intended use:** "AI-assisted quality assurance for radiology impression synthesis, requiring clinician confirmation before any clinical action."
