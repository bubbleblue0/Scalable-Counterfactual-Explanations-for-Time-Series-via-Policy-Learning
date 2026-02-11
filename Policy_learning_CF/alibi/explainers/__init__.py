"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .cfrl_base import CounterfactualRL
from .cfrl_tabular import CounterfactualRLTabular

__all__ = [
           "CounterfactualRL",
           "CounterfactualRLTabular"
           ]

try:
    from .shap_wrappers import KernelShap, TreeShap

    __all__ += ["KernelShap", "TreeShap"]
except ImportError:
    pass
