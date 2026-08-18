"""Les 7 sections du pipeline CORTEX : T, CO, RF, I, INV, CA, E."""

from cortex.sections.traducteur import TraducteurEntree, TraducteurSortie
from cortex.sections.comprehension import Comprehension
from cortex.sections.reflexion import Reflexion
from cortex.sections.imagination import Imagination
from cortex.sections.invention import Invention
from cortex.sections.comparaison import Comparaison
from cortex.sections.evaluation import Evaluation

__all__ = [
    "TraducteurEntree",
    "TraducteurSortie",
    "Comprehension",
    "Reflexion",
    "Imagination",
    "Invention",
    "Comparaison",
    "Evaluation",
]
