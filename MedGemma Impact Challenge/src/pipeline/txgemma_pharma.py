"""
TxGemma Pharmacovigilance Module.

After MedGemma generates clinical impressions, detected conditions are
mapped to drug SMILES structures and assessed for toxicity via
TxGemma-2B-Predict using the ClinTox benchmark format (TDC).

Pipeline: condition → drug → SMILES → ClinTox prompt → TxGemma prediction
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Condition → Drug mapping (common CXR-related medications) ──
CONDITION_DRUGS: dict[str, list[str]] = {
    "effusion": ["furosemide"],
    "pleural_effusion": ["furosemide"],
    "pneumonia": ["levofloxacin", "amoxicillin"],
    "consolidation": ["levofloxacin"],
    "cardiomegaly": ["lisinopril", "metoprolol"],
    "pulmonary_edema": ["furosemide", "nitroglycerin"],
    "atelectasis": ["albuterol"],
    "copd": ["tiotropium", "fluticasone"],
}

# ── Drug → SMILES structures ──
DRUG_SMILES: dict[str, str] = {
    "furosemide": "O=C(O)c1cc(S(=O)(=O)N)c(Cl)cc1NCc1ccco1",
    "levofloxacin": "C[C@@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@@H]1C(=O)O",
    "lisinopril": "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
    "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "albuterol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "nitroglycerin": "O=[N+]([O-])OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
    "tiotropium": "O=C(OC1CC2CCC(C1)[N+]2(C)C(c1cccs1)c1cccs1)C(O)(c1cccs1)c1cccs1",
    "fluticasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(OC(=O)SCF)C(=O)COC(=O)C(F)(F)F",
}


@dataclass
class TxGemmaResult:
    """Result of TxGemma ClinTox toxicity prediction for one drug."""

    drug: str
    smiles: str
    condition: str
    prediction: str  # "Yes" (toxic) or "No" (not toxic)
    confidence: float = 0.0
    inference_time_sec: float = 0.0


@dataclass
class PharmacovigilanceResult:
    """Aggregated pharmacovigilance results for a single case."""

    case_id: str
    conditions: list[str] = field(default_factory=list)
    drug_results: list[TxGemmaResult] = field(default_factory=list)
    any_toxic: bool = False
    n_drugs_tested: int = 0


class TxGemmaPharmacovigilance:
    """TxGemma-2B-Predict pharmacovigilance module.

    Maps detected clinical conditions to drug SMILES and runs
    ClinTox-format toxicity predictions via TxGemma.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        condition_drugs: Optional[dict] = None,
        drug_smiles: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.condition_drugs = condition_drugs or CONDITION_DRUGS
        self.drug_smiles = drug_smiles or DRUG_SMILES
        self.call_count = 0

    def assess_case(
        self,
        case_id: str,
        conditions: list[str],
    ) -> PharmacovigilanceResult:
        """Run pharmacovigilance assessment for detected conditions.

        Args:
            case_id: Unique case identifier.
            conditions: Clinical conditions detected by Diagnostician.

        Returns:
            PharmacovigilanceResult with per-drug toxicity predictions.
        """
        drug_results = []
        tested_drugs = set()

        for condition in conditions:
            cond_norm = condition.lower().replace(" ", "_")
            drugs = self.condition_drugs.get(cond_norm, [])

            for drug in drugs:
                if drug in tested_drugs:
                    continue
                tested_drugs.add(drug)

                smiles = self.drug_smiles.get(drug)
                if not smiles:
                    logger.warning(f"No SMILES for drug: {drug}")
                    continue

                result = self._predict_toxicity(drug, smiles, condition)
                drug_results.append(result)

        any_toxic = any(r.prediction.lower() == "yes" for r in drug_results)

        logger.info(
            f"[TxGemma] {case_id} | conditions={len(conditions)} | "
            f"drugs_tested={len(drug_results)} | toxic={any_toxic}"
        )

        return PharmacovigilanceResult(
            case_id=case_id,
            conditions=conditions,
            drug_results=drug_results,
            any_toxic=any_toxic,
            n_drugs_tested=len(drug_results),
        )

    def _predict_toxicity(
        self, drug: str, smiles: str, condition: str
    ) -> TxGemmaResult:
        """Run TxGemma ClinTox prediction for a single drug."""
        t0 = time.perf_counter()
        self.call_count += 1

        prompt = (
            f"Given the drug SMILES string: {smiles}\n"
            f"Predict whether this drug is toxic in clinical trials.\n"
            f"Answer with Yes or No."
        )

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=10)
            prediction = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        else:
            prediction = "No"  # Fallback when model not loaded

        return TxGemmaResult(
            drug=drug,
            smiles=smiles,
            condition=condition,
            prediction=prediction,
            inference_time_sec=time.perf_counter() - t0,
        )
