"""Pipeline orchestration: DAG execution, HITL gate, and pharmacovigilance."""

# Lazy imports to avoid requiring torch at import time
def __getattr__(name):
    if name == "SentinelPipeline":
        from src.pipeline.sentinel import SentinelPipeline
        return SentinelPipeline
    if name == "HITLGate":
        from src.pipeline.hitl_gate import HITLGate
        return HITLGate
    if name == "TxGemmaPharmacovigilance":
        from src.pipeline.txgemma_pharma import TxGemmaPharmacovigilance
        return TxGemmaPharmacovigilance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SentinelPipeline", "HITLGate", "TxGemmaPharmacovigilance"]
