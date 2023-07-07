from .BaseLoss import BaseLoss
from .SoftmaxLoss import SoftmaxLoss
from .CosineSimilarityLoss import CosineSimilarityLoss
from .DenoisingAutoEncoderLoss import DenoisingAutoEncoderLoss
from .MultipleNegativesRankingLoss import MultipleNegativesRankingLoss

# TODO: add mapper

NAME2LOSS_MAP = {
    "multi-neg-ranking": MultipleNegativesRankingLoss,
    "softmax": SoftmaxLoss,
    "denoising-autoencoder": DenoisingAutoEncoderLoss,
    "cosine-similarity": CosineSimilarityLoss
}