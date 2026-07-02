from collections.abc import Callable

import numpy as np
from PIL import Image

from .._base_classes import SimilarityMetric
from .._utils import cosine_similarity
from ..lazy_import import OptionalImport
from ..typing import MatLike

with OptionalImport(package="torch", extra="nn") as _torch_import:
    import torch
    from torchvision import transforms

_torch_import.check()

device = "cuda" if torch.cuda.is_available() else "cpu"


# Siamese Neural Network implementation with PyTorch
# Initializing base parent class and move model to device
class SiameseNeuralNetwork(torch.nn.Module, SimilarityMetric):
    def __init__(
        self,
        backbone: torch.nn.Module,
        embedding_dim: int,
        similarity_func: Callable[[MatLike, MatLike], MatLike] = cosine_similarity,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self._backbone = backbone
        self._head = torch.nn.Linear(int(backbone.output_dim), embedding_dim)
        self.similarity_func = similarity_func
        self.to(self.device)

    def forward(self, x) -> torch.Tensor:
        """
        forward() function will returns tensor of specified image
        """
        features = self._backbone(x)
        embeddings = self._head(features)
        # L2 normalization on embeddings
        # It helps to make cosine similarity via dot product
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings

    # User may pass images in multiple formats, but preprocess should return pytorch tensor for network
    def preprocess(
        self, image, dims: str = "HWC", value_range: tuple[float, float] = (0, 255)
    ) -> torch.Tensor:
        """
        Preprocess an image into model-ready tensor.

        Args:
            image:        image
            dims:         Channel layout for numpy input — "HWC" (H×W×C) or "CHW" (C×H×W)
            value_range:  (min, max) pixel value range; used to rescale to [0,1] before standard ImageNet normalization

        """

        # "if not isinstance(image, Image.Image):" This asks, "is the image not a PIL image ?

        # Let say img = Image.open("cat.jpg")
        # then isinstance(img, Image.Image) returns True
        # so the "if" block is skipped.
        # else it goes into "if" block

        if not isinstance(image, Image.Image):
            arr = np.asarray(image, dtype=np.float32)  # Converting into numpy float,

            # Transpose CHW to HWC before PIL conversion, since Image.fromarray expects HWC layout
            if dims.upper() == "CHW":
                arr = arr.transpose(1, 2, 0)  # CHW to HWC

            # Normalization requires floating point operation
            lo, hi = value_range
            # Normalize the array into [0, 1] (standardization)
            arr = (arr - lo) / (hi - lo + 1e-8)
            # Clipping invalid values (who's normalized not between 0, 1)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)  # Converting back to uint8 image
            # After normalization values always between [0, 1]
            # So, we need to make it again between [0, 255] for tensor operation
            image = Image.fromarray(arr)  # Converting back to PIL image

        # Converting back to RGB image for tensor operation
        image = image.convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform(image)

    @torch.no_grad()
    def encode(
        self,
        image: MatLike,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0, 255),
    ):
        # For encoding image, we need to switch into eval mode so that BatchNorm/ Dropout hevaes correctly during inference
        # Also we're restoring previous training state afterwards so that calling encode() mind-training doesn't break the training loop
        was_training = self.training
        self.eval()
        try:
            # first we're doing preprocessing of given image
            x = self.preprocess(image, dims=dims, value_range=value_range)
            x = x.unsqueeze(0).to(self.device)
            embedding = self.forward(x)
            return embedding.cpu().numpy().squeeze()

        finally:
            if was_training:
                self.train()

    def similarity_score(
        self,
        image_a,
        image_b,
        dims_a="HWC",
        dims_b="HWC",
        value_range: tuple[float, float] = (0, 255),
    ):
        emb_a = self.encode(image=image_a, dims=dims_a, value_range=value_range)
        emb_b = self.encode(image=image_b, dims=dims_b, value_range=value_range)
        score = self.similarity_func(emb_a, emb_b)
        return float(score)

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    @head.setter
    def head(self, new_head: torch.nn.Module) -> None:
        if not isinstance(new_head, torch.nn.Module):
            raise ValueError(
                f"Expected new_head to be an instance of torch.nn.Module, got {type(new_head)}"
            )

        # in_features only exists on nn.Linear
        # for custom using getattr
        in_features = getattr(new_head, "in_features", None)
        if in_features is not None and in_features != self._backbone.output_dim:
            raise ValueError(
                f"Expected new_head to have in_features equal to {self._backbone.output_dim}, got {new_head.in_features}"
            )
        self._head = new_head


# To make prediction more good, we need to make very less error between two images
# https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf (original siamese neural network paper for reference)
# https://arxiv.org/abs/2004.11362 (contrastive loss)
# Small distance = Similar Images
# Large distance = Different Images


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b)
        loss = 0.5 * (
            labels * distance.pow(2)
            + (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        )
        return loss.mean()
