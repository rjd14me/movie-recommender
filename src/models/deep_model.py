from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset + MF model
# -----------------------------


class RatingsDataset(Dataset):
    """
    Simple dataset wrapping user indices, movie indices, and rating values.
    """

    def __init__(self, user_ids: np.ndarray, movie_ids: np.ndarray, ratings: np.ndarray):
        # Expect: int64 for ids, float32 for ratings
        self.user_ids = user_ids.astype("int64")
        self.movie_ids = movie_ids.astype("int64")
        self.ratings = ratings.astype("float32")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx):
        # Return numpy scalars with correct dtypes so DataLoader -> torch tensors
        return (
            np.int64(self.user_ids[idx]),
            np.int64(self.movie_ids[idx]),
            np.float32(self.ratings[idx]),
        )


class MFModel(nn.Module):
    """
    Simple matrix factorization model:

        r_hat(u, i) = dot(U[u], V[i]) + b_u + b_i + b_global

    where U and V are user/movie embeddings, and b are biases.
    """

    def __init__(self, n_users: int, n_movies: int, n_factors: int = 64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize embeddings with small random values
        nn.init.normal_(self.user_factors.weight, 0, 0.05)
        nn.init.normal_(self.movie_factors.weight, 0, 0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        u_vec = self.user_factors(user_idx)              # (batch, n_factors)
        m_vec = self.movie_factors(movie_idx)            # (batch, n_factors)
        u_b = self.user_bias(user_idx).squeeze(-1)       # (batch,)
        m_b = self.movie_bias(movie_idx).squeeze(-1)     # (batch,)

        dot = (u_vec * m_vec).sum(dim=-1)                # (batch,)
        return dot + u_b + m_b + self.global_bias        # (batch,)


# -----------------------------
# Config + main recommender
# -----------------------------


@dataclass
class DeepRecommenderConfig:
    """
    Hyperparameters for the DeepRecommender.
    """
    n_factors: int = 64
    n_epochs: int = 5
    batch_size: int = 2048
    lr: float = 5e-3
    weight_decay: float = 1e-6
    min_rating_count: int = 5  # filter out movies with too few ratings


class DeepRecommender:
    """
    Wrapper around MFModel that:

    - builds user/movie index mappings
    - trains the model on ratings
    - stores learned movie embeddings
    - exposes similar_movies(movie_id, n) based on cosine similarity
    """

    def __init__(
        self,
        config: Optional[DeepRecommenderConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DeepRecommenderConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[MFModel] = None

        # ID <-> index mappings
        self.user_id_to_index: Dict[int, int] = {}
        self.movie_id_to_index: Dict[int, int] = {}
        self.index_to_movie_id: Dict[int, int] = {}

        # Movie metadata (with stats)
        self.movies: Optional[pd.DataFrame] = None

        # Learned embeddings for movies: shape (n_movies, n_factors)
        self.movie_embeddings: Optional[np.ndarray] = None

    # ---------- training ----------

    def fit(self, ratings: pd.DataFrame, movies_with_stats: pd.DataFrame) -> None:
        """
        Train the MF model on ratings and store movie embeddings.

        Parameters
        ----------
        ratings : DataFrame
            Must contain at least: userId, movieId, rating
        movies_with_stats : DataFrame
            Movies dataframe with columns [movieId, title, genres, rating_mean, rating_count, ...]
        """
        cfg = self.config

        # Optionally filter out rarely-rated movies
        counts = ratings["movieId"].value_counts()
        keep_movie_ids = counts[counts >= cfg.min_rating_count].index

        filtered = ratings[ratings["movieId"].isin(keep_movie_ids)].copy()
        if filtered.empty:
            raise ValueError("No ratings left after filtering by min_rating_count.")

        # Build id -> index mappings
        unique_user_ids = np.sort(filtered["userId"].unique())
        unique_movie_ids = np.sort(filtered["movieId"].unique())

        self.user_id_to_index = {uid: idx for idx, uid in enumerate(unique_user_ids)}
        self.movie_id_to_index = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
        self.index_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_index.items()}

        user_idx = filtered["userId"].map(self.user_id_to_index).values.astype("int64")
        movie_idx = filtered["movieId"].map(self.movie_id_to_index).values.astype("int64")
        ratings_vals = filtered["rating"].values.astype("float32")

        dataset = RatingsDataset(user_idx, movie_idx, ratings_vals)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        n_users = len(unique_user_ids)
        n_movies = len(unique_movie_ids)

        self.model = MFModel(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=cfg.n_factors,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(cfg.n_epochs):
            total_loss = 0.0
            n_batches = 0

            for u, m, r in loader:
                # Ensure correct dtypes
                u = u.to(self.device).long()
                m = m.to(self.device).long()
                r = r.to(self.device).float()  # float32 ratings

                optimizer.zero_grad()
                preds = self.model(u, m).float()
                loss = loss_fn(preds, r)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"[DeepRecommender] Epoch {epoch+1}/{cfg.n_epochs} - train MSE: {avg_loss:.4f}")

        # Extract movie embeddings to CPU numpy array
        with torch.no_grad():
            self.model.eval()
            movie_factors = self.model.movie_factors.weight.detach().cpu().numpy()
        self.movie_embeddings = movie_factors

        # Store movie metadata (with stats)
        self.movies = movies_with_stats.copy()

    # ---------- similarity ----------

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return top-n similar movies using cosine similarity on learned embeddings.

        Parameters
        ----------
        movie_id : int
            The MovieLens movieId to find neighbours for.
        n : int
            Number of similar movies to return.

        Returns
        -------
        DataFrame
            Columns: [movie_index, similarity, movieId, title, genres, rating_mean, rating_count, ...]
        """
        if self.movie_embeddings is None or self.movies is None:
            raise RuntimeError("Model not fitted or embeddings/movies not available.")
        if movie_id not in self.movie_id_to_index:
            raise KeyError(f"movie_id {movie_id} not present in trained model.")

        movie_idx = self.movie_id_to_index[movie_id]
        emb = self.movie_embeddings

        target_vec = emb[movie_idx]
        # cosine similarity = dot(a,b) / (||a|| * ||b||)
        dot = emb @ target_vec
        norms = np.linalg.norm(emb, axis=1) * np.linalg.norm(target_vec)
        sims = dot / (norms + 1e-8)

        sims_df = pd.DataFrame(
            {"movie_index": np.arange(emb.shape[0]), "similarity": sims}
        )

        # remove the movie itself
        sims_df = sims_df[sims_df["movie_index"] != movie_idx]

        # map back to movieId
        sims_df["movieId"] = sims_df["movie_index"].map(self.index_to_movie_id)

        # top-n by similarity
        sims_df = sims_df.sort_values("similarity", ascending=False).head(n)

        # join with movie metadata
        sims_df = sims_df.merge(
            self.movies,
            on="movieId",
            how="left",
        )

        return sims_df.reset_index(drop=True)

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        """
        Save embeddings + mappings to disk (NOT the full MF model).
        """
        if self.movie_embeddings is None:
            raise RuntimeError("No embeddings to save; train the model first.")

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
        """
        Load a saved DeepRecommender (embeddings + mappings) from disk.

        The full MFModel isn't needed at inference time because we only use
        the movie embeddings for similarity search.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved deep model at {path.resolve()}")

        payload = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=False,  # important for PyTorch 2.6+
        )

        cfg = DeepRecommenderConfig(**payload["config"])
        rec = cls(config=cfg, device=device)

        rec.movie_embeddings = payload["movie_embeddings"]
        rec.movie_id_to_index = payload["movie_id_to_index"]
        rec.index_to_movie_id = payload["index_to_movie_id"]
        rec.movies = movies_with_stats.copy()

        return rec
