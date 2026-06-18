
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from siamese_neural_network import SiameseNeuralNetwork, ContrastiveLoss
from pyvisim.datasets import OxfordFlowerDataset

# For reproducibility of the code
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# For siamese neural network, we need a backbone that can encode the image
# For the following we're using resenet 18 backbone
# For more, visit https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

class ResNetBackbone(nn.Module):
    """Resnet-18 with final FC layer removed; exposes output_dim"""

    def __init__(self, pretrained: bool=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None # getting the default ResNet-18 model training weights
        base = models.resnet18(weights = weights)
        # dropping the final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.output_dim = base.fc.in_features # output_dim by the backbone
        # 512 for the case of ResNet 18

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1) # output dim (B, 512)
    

# Dataset returns uint8 HWC numpy arrays, so we convert via PIL
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),                        # uint8 numpy → PIL
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),                          # [0,255] → [0,1] float
    _NORMALIZE,
])

VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _NORMALIZE,
])
   

# For contrastive loss, we need both positive & negative pair to make it a label
# https://arxiv.org/abs/2004.11362 (contrastive loss)
class OxfordSiamesePairs(Dataset):
    """
    Wraps OxfordFlowerDataset and yields contrastive pairs on-the-fly.
 
    Each __getitem__ returns (img_a_tensor, img_b_tensor, label) where:
      label = 1 : same flower class   (positive pair)
      label = 0 : different class     (negative pair)
 
    Transforms are applied here since OxfordFlowerDataset doesn't support
    them internally yet.
    """

    def __init__(self, purpose: str, transform, pos_fraction: float = 0.5):
        # Load the underlying dataset
        # Currenly no transform is needed, we'll do it in _get_tensor()
        self._base = OxfordFlowerDataset(transform=None, purpose=purpose)
        self.transform = transform
        self.pos_fraction = pos_fraction

        # Build label : list[index] map for fast pair mining
        self._label_to_indices: dict[int, list[int]] = {}
        for idx in range(len(self._base)):
            _, label, _ = self._base[idx]
            self._label_to_indices.setdefault(label, []).append(idx)

        self._labels_list = list(self._label_to_indices.keys())
        assert len(self._labels_list) >= 2, "Need >= 2 classes for contrastive training."
    

    def __len__(self) -> int:
        return len(self._base)

    def _get_tensor(self, idx: int) -> torch.Tensor:
        image, _, _ = self._base[idx]           # uint8 numpy HWC
        return self.transform(image)
    
    def __getitem__(self, idx: int):
        _, label_a, _ = self._base[idx]
        img_a = self._get_tensor(idx)

        if random.random() < self.pos_fraction:
            # Positive: another image of the same class
            same_indices = [i for i in self._label_to_indices[label_a] if i != idx]
            idx_b = random.choice(same_indices) if same_indices else idx
            pair_label = 1
        else:
            # Negative: an image from a different class
            neg_label = random.choice([l for l in self._labels_list if l != label_a])
            idx_b = random.choice(self._label_to_indices[neg_label])
            pair_label = 0

        img_b = self._get_tensor(idx_b)
        return img_a, img_b, torch.tensor(pair_label, dtype=torch.float32)


# configurations for training model
CONFIG = dict(
    embedding_dim=128,
    margin=1.0,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=32,
    num_epochs=20,
    num_workers=4,
    checkpoint_dir="checkpoints", # for storing good performing model
    # once model is trained, we don't need it, we only need good performing model
    log_every_n=50,
)


# Training and evaluation loop for siamese nn training for each epoch
def run_epoch(model, loader, criterion, optimizer, is_train: bool, epoch: int) -> float:
    """"  
    Training & Evaluation loop for siamese nn training for one complete epoch.
    
    Args:
        model     : Siamese Neural Network
        loader    : DataLoader containing image pairs
        criterion : Loss function (Contrastive Loss for this case)
        optimizer : Optimizer function for nn
        is_train  : True -> training, False -> False
        epoch     : Current epoch number

    Output:
        float     : average loss over the epoch
    """
    model.train() if is_train else model.eval() # setting model mode
    total_loss = 0.0
    # choosing gradient context
    # during training torch.enable_grad(), during vlaidation torch.no_grad()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for step, (img_a, img_b, labels) in enumerate(loader):
            # Moving tensors to device
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)

            # Forward pass
            emb_a = model(img_a) # embedding of img a after forward pass
            emb_b = model(img_b) # embedding of img b after forward pass
            loss = criterion(emb_a, emb_b, labels) # contrastive loss between img a & img b

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to prevent explosive gradient
                optimizer.step()

            total_loss += loss.item()

            if is_train and (step + 1) % CONFIG["log_every_n"] == 0:
                print(f"  Epoch {epoch:03d} | step {step + 1:04d} " 
                      f"| loss {total_loss / (step + 1):.4f}")

    return total_loss / len(loader)

    

# Training functionality for siamese network
def train():
    import os
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True) # checkpoint directory for storing best model

    # Oxford train split = 6149 images 
    # Oxford val split   = 1020 images
    train_ds = OxfordSiamesePairs("train", transform=TRAIN_TF)
    val_ds = OxfordSiamesePairs("validation", transform=VAL_TF)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )

    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")
    print(f"Flower classes: {len(train_ds._labels_list)}")

    # loading backbone and model
    backbone = ResNetBackbone(pretrained=True)
    model = SiameseNeuralNetwork(
        backbone=backbone,
        embedding_dim=CONFIG["embedding_dim"],
        device=device,
    )

    # loss, optimizer & scheduler
    criterion = ContrastiveLoss(margin=CONFIG["margin"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )

    best_val = float("inf")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               is_train=True, epoch=epoch)
        val_loss = run_epoch(model, val_loader, criterion, optimizer,
                             is_train=False, epoch=epoch)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss, "cfg": CONFIG,
        }
        torch.save(ckpt, f"{CONFIG['checkpoint_dir']}/latest.pt")
        # saving the best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, f"{CONFIG['checkpoint_dir']}/best.pt")
            print(f"New best val loss: {best_val:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


# After training model
# Our mail goal is to find similarity between two images
# So, we'll run a quick testing to check if the network learns well about inputs or not
def demo(checkpoint: str, img_path_a: str, img_path_b: str):
    """Compare two flower images using a trained checkpoint."""

    from PIL import Image as PILImage
    backbone = ResNetBackbone(pretrained=False)
    model = SiameseNeuralNetwork(backbone=backbone,
                                 embedding_dim=CONFIG["embedding_dim"],
                                 device=device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    img_a = PILImage.open(img_path_a).convert("RGB")
    img_b = PILImage.open(img_path_b).convert("RGB")
    score = model.similarity_score(img_a, img_b)
    print(f"Similarity: {score:.4f}  (range −1 to 1, higher = more similar)")


if __name__ == "__main__":
    train()
