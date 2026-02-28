"""Three-agent adversarial architecture for clinical AI validation."""

# Lazy imports to avoid requiring torch at import time
def __getattr__(name):
    if name == "Diagnostician":
        from src.agents.diagnostician import Diagnostician
        return Diagnostician
    if name == "Challenger":
        from src.agents.challenger import Challenger
        return Challenger
    if name == "FactChecker":
        from src.agents.factchecker import FactChecker
        return FactChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Diagnostician", "Challenger", "FactChecker"]
