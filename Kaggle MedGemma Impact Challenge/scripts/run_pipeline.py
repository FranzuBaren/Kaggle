#!/usr/bin/env python3
"""
CXR-Sentinel — CLI Entry Point.

Run the full three-agent adversarial validation pipeline from the command line.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml --data-dir /path/to/openi
    python scripts/run_pipeline.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.diagnostician import Diagnostician
from src.agents.challenger import Challenger
from src.agents.factchecker import FactChecker
from src.pipeline.sentinel import SentinelPipeline, CaseInput
from src.pipeline.hitl_gate import HITLGate
from src.pipeline.txgemma_pharma import TxGemmaPharmacovigilance
from src.utils.clinical_vocab import ClinicalVocabulary
from src.utils.data_loader import OpenIDataLoader
from src.utils.model_loader import MedGemmaEngine
from src.utils.metrics import bootstrap_ci


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CXR-Sentinel: Multi-Agent Adversarial Clinical AI Validation",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--n-cases", type=int, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-txgemma", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("cxr-sentinel")

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  CXR-Sentinel v1.5 — Multi-Agent Adversarial Validation")
    logger.info("=" * 70)

    t_start = time.perf_counter()

    # Stage 1: Load data
    logger.info("[1/6] Loading OpenI dataset...")
    loader = OpenIDataLoader(args.data_dir)
    all_cases = loader.load_reports()
    n_per_cat = args.n_cases or config["data"]["n_per_category"]
    cohort = loader.select_cohort(all_cases, n_per_category=n_per_cat)

    # Stage 2: Load models
    logger.info("[2/6] Loading MedGemma 1.5 4B...")
    engine = MedGemmaEngine(
        model_id=config["model"]["medgemma"]["model_id"],
        quantize=config["model"]["medgemma"]["quantize"] and not args.no_gpu,
    )
    engine.load()

    # Stage 3: Initialize agents
    logger.info("[3/6] Initializing agents...")
    vocab = ClinicalVocabulary()
    diagnostician = Diagnostician(engine=engine, vocab=vocab)
    challenger = Challenger(engine=engine, vocab=vocab)
    factchecker = FactChecker(vocab=vocab)
    hitl_gate = HITLGate(
        hi_threshold=config["hitl"]["hi_threshold"],
        pss_threshold=config["hitl"]["pss_threshold"],
        sce_threshold=config["hitl"]["sce_threshold"],
    )
    pipeline = SentinelPipeline(diagnostician, challenger, factchecker, hitl_gate)

    # Stage 4: Run pipeline
    logger.info("[4/6] Running pipeline...")
    case_inputs = [
        CaseInput(
            case_id=c.case_id,
            findings=c.findings,
            indication=c.indication,
            gt_impression=c.impression,
            gt_severity=c.ground_truth_severity.name.lower(),
        )
        for c in cohort
    ]
    results = pipeline.run_batch(case_inputs)
    stats = pipeline.get_summary_stats(results)

    # Stage 5: Report
    logger.info("[5/6] Results:")
    ok_results = [r for r in results if r.pipeline_status == "OK"]
    dcr_vals = [r.factchecker_result.dcr for r in ok_results]
    hi_vals = [r.factchecker_result.hallucination_index for r in ok_results]
    pss_vals = [r.challenger_result.mean_pss for r in ok_results]

    dcr_mean, dcr_lo, dcr_hi = bootstrap_ci(dcr_vals)
    hi_mean, hi_lo, hi_hi = bootstrap_ci(hi_vals)
    pss_mean, pss_lo, pss_hi = bootstrap_ci(pss_vals)

    logger.info(f"  DCR: {dcr_mean:.1%} [{dcr_lo:.2f}, {dcr_hi:.2f}]")
    logger.info(f"  HI:  {hi_mean:.1%} [{hi_lo:.2f}, {hi_hi:.2f}]")
    logger.info(f"  PSS: {pss_mean:.1%} [{pss_lo:.2f}, {pss_hi:.2f}]")
    logger.info(f"  F1:  {stats['mean_f1']:.3f}")
    logger.info(f"  HITL: {stats['hitl_rate']:.0%}")

    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": stats}, f, indent=2, default=str)
    logger.info(f"  Saved to {results_path}")

    # Stage 6: TxGemma
    if not args.no_txgemma:
        logger.info("[6/6] Running TxGemma pharmacovigilance...")
        pharma = TxGemmaPharmacovigilance()
        for r in ok_results:
            pharma.assess_case(r.case_id, r.diagnostician_result.entities)

    elapsed = time.perf_counter() - t_start
    logger.info(f"\nTotal: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
