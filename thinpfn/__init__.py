"""ThinPFN: a prior-fitted network for small-sample tabular classification, pretrained on synthetic tables and validated on
public credit portfolios. Scores a new table from a few hundred labelled rows in one forward pass, no training."""
from .classifier import ThinPFNClassifier, load_pretrained
from .model import PFNConfig, ThinPFN

__version__ = "1.1.0"
__all__ = ["ThinPFNClassifier", "load_pretrained", "ThinPFN", "PFNConfig"]
