import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MorphologyDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        parent_masks = data["parent_masks"].astype(np.float32)
        child_masks = data["child_masks"].astype(np.float32)
        fitness_delta = data["fitness_delta"].astype(np.float32)
        parent_fitness = data["parent_fitness"].astype(np.float32)
        edit_distance = data["edit_distance"].astype(np.float32)
        parent_id = data["parent_id"].astype(np.int64)

        if parent_masks.ndim != 4:
            raise ValueError(
                f"Expected parent_masks shape (N, C, H, W), got {parent_masks.shape}"
            )
        if child_masks.ndim != 4:
            raise ValueError(
                f"Expected child_masks shape (N, C, H, W), got {child_masks.shape}"
            )

        n = parent_masks.shape[0]
        if (
            child_masks.shape[0] != n
            or fitness_delta.shape[0] != n
            or parent_fitness.shape[0] != n
            or edit_distance.shape[0] != n
            or parent_id.shape[0] != n
        ):
            raise ValueError("Dataset arrays must all have the same first dimension.")

        finite_mask = (
            np.isfinite(fitness_delta)
            & np.isfinite(parent_fitness)
            & np.isfinite(edit_distance)
            & np.isfinite(parent_masks).reshape(n, -1).all(axis=1)
            & np.isfinite(child_masks).reshape(n, -1).all(axis=1)
        )

        num_bad = int((~finite_mask).sum())
        if num_bad > 0:
            print(f"[MorphologyDataset] Filtering out {num_bad} invalid samples out of {n}.")

        parent_masks = parent_masks[finite_mask]
        child_masks = child_masks[finite_mask]
        fitness_delta = fitness_delta[finite_mask]
        parent_fitness = parent_fitness[finite_mask]
        edit_distance = edit_distance[finite_mask]
        parent_id = parent_id[finite_mask]

        self.parent_masks = torch.from_numpy(parent_masks).float()
        self.child_masks = torch.from_numpy(child_masks).float()
        self.fitness_delta = torch.from_numpy(fitness_delta).float()
        self.parent_fitness = torch.from_numpy(parent_fitness).float()
        self.edit_distance = torch.from_numpy(edit_distance).float()
        self.parent_id = torch.from_numpy(parent_id).long()

    def __len__(self) -> int:
        return self.parent_masks.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "parent_mask": self.parent_masks[idx],
            "child_mask": self.child_masks[idx],
            "fitness_delta": self.fitness_delta[idx],
            "parent_fitness": self.parent_fitness[idx],
            "edit_distance": self.edit_distance[idx],
            "parent_id": self.parent_id[idx],
        }


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(1)
        z = self.proj(h)
        return z


class MutationRankerRegressor(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim)

        self.head = nn.Sequential(
            nn.Linear(3 * latent_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        parent_mask: torch.Tensor,
        child_mask: torch.Tensor,
        parent_fitness: torch.Tensor,
        edit_distance: torch.Tensor,
    ) -> torch.Tensor:
        z_parent = self.encoder(parent_mask)
        z_child = self.encoder(child_mask)
        z_diff = z_child - z_parent

        scalar_feats = torch.stack([parent_fitness, edit_distance], dim=1)
        feat = torch.cat([z_parent, z_child, z_diff, scalar_feats], dim=1)

        score = self.head(feat).squeeze(1)
        return score