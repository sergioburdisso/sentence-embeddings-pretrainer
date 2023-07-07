import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)


class BaseLoss(nn.Module):
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        pass