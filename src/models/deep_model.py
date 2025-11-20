from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class RatingsDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, movie_ids: np.ndarray, ratings: np.ndarray):
        self.user_ids = user_ids.astype("int64")
        self.movie_ids = movie_ids.astype("int64")
        self.ratings = ratings.astype("float32")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            np.int64(self.user_ids[idx]),
            np.int64(self.movie_ids[idx]),
            np.float32(self.ratings[idx]),
        )


class MFModel(nn.Module):
    def __init__(self, n_users: int, n_movies: int, n_factors: int = 64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_factors.weight, 0.0, 0.05)
        nn.init.normal_(self.movie_factors.weight, 0.0, 0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        u_vec = self.user_factors(user_idx)
        m_vec = self.movie_factors(movie_idx)
        u_b = self.user_bias(user_idx).squeeze(-1)
        m_b = self.movie_bias(movie_idx).squeeze(-1)
        dot = (u_vec * m_vec).sum(dim=-1)
        return dot + u_b + m_b + self.global_bias


@dataclass
class DeepRecommenderConfig:
    n_factors: int = 64
    n_epochs: int = 5
    batch_size: int = 2048
    lr: float = 5e-3
    weight_decay: float = 1e-6
    min_rating_count: int = 5


class DeepRecommender:
    def __init__(
        self,
        config: Optional[DeepRecommenderConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DeepRecommenderConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[MFModel] = None

        self.user_id_to_index: Dict[int, int] = {}
        self.movie_id_to_index: Dict[int, int] = {}
        self.index_to_movie_id: Dict[int, int] = {}

        self.movies: Optional[pd.DataFrame] = None
        self.movie_embeddings: Optional[np.ndarray] = None

    def fit(self, ratings: pd.DataFrame, movies_with_stats: pd.DataFrame) -> None:
        cfg = self.config

        counts = ratings["movieId"].value_counts()
        keep_ids = counts[counts >= cfg.min_rating_count].index
        filtered = ratings[ratings["movieId"].isin(keep_ids)].copy()
        if filtered.empty:
            raise ValueError("No ratings left after filtering by min_rating_count.")

        unique_user_ids = np.sort(filtered["userId"].unique())
        unique_movie_ids = np.sort(filtered["movieId"].unique())

        self.user_id_to_index = {uid: i for i, uid in enumerate(unique_user_ids)}
        self.movie_id_to_index = {mid: i for i, mid in enumerate(unique_movie_ids)}
        self.index_to_movie_id = {i: mid for mid, i in self.movie_id_to_index.items()}

        user_idx = filtered["userId"].map(self.user_id_to_index).values.astype("int64")
        movie_idx = filtered["movieId"].map(self.movie_id_to_index).values.astype("int64")
        rating_vals = filtered["rating"].values.astype("float32")

        dataset = RatingsDataset(user_idx, movie_idx, rating_vals)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        n_users = len(unique_user_ids)
        n_movies = len(unique_movie_ids)

        self.model = MFModel(n_users, n_movies, cfg.n_factors).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(cfg.n_epochs):
            total_loss = 0.0
            num_batches = 0

            for u, m, r in loader:
                u = u.to(self.device).long()
                m = m.to(self.device).long()
                r = r.to(self.device).float()

                optimizer.zero_grad()
                preds = self.model(u, m).float()
                loss = loss_fn(preds, r)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"[DeepRecommender] Epoch {epoch+1}/{cfg.n_epochs} - train MSE: {avg_loss:.4f}")

        with torch.no_grad():
            self.model.eval()
            emb = self.model.movie_factors.weight.detach().cpu().numpy()
        self.movie_embeddings = emb
        self.movies = movies_with_stats.copy()

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if self.movie_embeddings is None or self.movies is None:
            raise RuntimeError("Model not fitted yet.")
        if movie_id not in self.movie_id_to_index:
            raise KeyError(f"movie_id {movie_id} not found in model.")

        idx = self.movie_id_to_index[movie_id]
        emb = self.movie_embeddings

        target_vec = emb[idx]
        target_norm = float(np.linalg.norm(target_vec))

        sims_list = []
        for i in range(emb.shape[0]):
            if i == idx:
                continue
            v = emb[i]
            num = float(np.dot(v, target_vec))
            denom = float(np.linalg.norm(v) * (target_norm + 1e-8))
            if denom == 0.0:
                sim = 0.0
            else:
                sim = num / denom
            sims_list.append((i, sim))

        sims_list.sort(key=lambda x: x[1], reverse=True)
        top = sims_list[:n]

        rows = []
        for movie_index, similarity in top:
            mid = self.index_to_movie_id.get(movie_index)
            rows.append((movie_index, mid, similarity))

        sims_df = pd.DataFrame(rows, columns=["movie_index", "movieId", "similarity"])
        sims_df = sims_df.merge(self.movies, on="movieId", how="left")

        return sims_df.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        if self.movie_embeddings is None:
            raise RuntimeError("No embeddings to save.")
        payload = {
            "config": self.config.__dict__,
            "movie_embeddings": self.movie_embeddings,
            "movie_id_to_index": self.movie_id_to_index,
            "index_to_movie_id": self.index_to_movie_id,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        movies_with_stats: pd.DataFrame,
        device: Optional[str] = None,
    ) -> "DeepRecommender":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path.resolve()}")

        payload = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=False,
        )

        cfg = DeepRecommenderConfig(**payload["config"])
        rec = cls(config=cfg, device=device)

        rec.movie_embeddings = payload["movie_embeddings"]
        rec.movie_id_to_index = payload["movie_id_to_index"]
        rec.index_to_movie_id = payload["index_to_movie_id"]
        rec.movies = movies_with_stats.copy()

        return rec
