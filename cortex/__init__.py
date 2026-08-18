"""
CORTEX — Modèle d'intelligence artificielle fractal inspiré du cerveau humain.

Architecture fractale en 5 niveaux (3 125 modules) combinant
des transformeurs décodeurs imbriqués avec des structures
neuronales inspirées du cerveau (créativité, visualisation, conception).
"""

__version__ = "0.1.0"

# Import protégé : tant que tous les sous-modules (cortex.sections,
# cortex.crvg, ...) ne sont pas fournis, on ne veut pas empêcher le
# reste du package (ex. cortex.tokenizer) de se charger normalement.
try:
    from cortex.config import CortexConfig
    from cortex.model import CortexModel
    __all__ = ["CortexConfig", "CortexModel"]
except ImportError:
    CortexConfig = None
    CortexModel = None
    __all__ = []