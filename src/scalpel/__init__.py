"""Scalpel: surgical model editing and bias mitigation platform.

Scalpel isolates latent bias circuits in foundation models using mechanistic
interpretability (difference-of-means concept directions + per-component
attribution) and permanently removes them with closed-form rank-one weight
edits, while measuring commercial performance retention.
"""

__version__ = "0.3.0"

from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import DebiasResult, run_debias_pipeline

__all__ = ["DebiasResult", "SurgeryConfig", "__version__", "run_debias_pipeline"]
