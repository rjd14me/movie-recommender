import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MovieLensGenreDataset(Dataset):
    def __init__(self, user_indices, movie_indices, ratings, genre_matrix):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.movie_indices = torch.tensor(movie_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.genre_matrix = torch.tensor(genre_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        movie_idx = self.movie_indices[idx]
        return (
            self.user_indices[idx],
            self.genre_matrix[movie_idx],
            self.ratings[idx],
        )


class GenreRecommender(nn.Module):

    def __init__(self, num_users, num_genres, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.genre_projection = nn.Sequential(
            nn.Linear(num_genres, embedding_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_ids, genre_vectors):
        user_vec = self.user_embedding(user_ids)
        genre_vec = self.genre_projection(genre_vectors)
        combined = torch.cat([user_vec, genre_vec], dim=1)
        rating_pred = self.regressor(combined)
        return rating_pred.squeeze(1)

    def genre_embedding(self, genre_vectors):
        return self.genre_projection(genre_vectors)


def train_genre_model(
    train_users,
    train_movies,
    train_ratings,
    val_users,
    val_movies,
    val_ratings,
    genre_matrix,
    num_users,
    num_genres,
    epochs=5,
    batch_size=256,
    lr=1e-3,
    device=None,
):
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = MovieLensGenreDataset(train_users, train_movies, train_ratings, genre_matrix)
    val_dataset = MovieLensGenreDataset(val_users, val_movies, val_ratings, genre_matrix)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GenreRecommender(num_users=num_users, num_genres=num_genres).to(resolved_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for user_ids, genre_vecs, ratings in train_loader:
            user_ids = user_ids.to(resolved_device)
            genre_vecs = genre_vecs.to(resolved_device)
            ratings = ratings.to(resolved_device)

            optimizer.zero_grad()
            preds = model(user_ids, genre_vecs)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for user_ids, genre_vecs, ratings in val_loader:
            user_ids = user_ids.to(resolved_device)
            genre_vecs = genre_vecs.to(resolved_device)
            ratings = ratings.to(resolved_device)
            preds = model(user_ids, genre_vecs)
            val_losses.append(criterion(preds, ratings).item())

    metrics = {"val_mse": float(np.mean(val_losses)) if val_losses else float("nan")}
    return model.to("cpu"), metrics


def compute_genre_embeddings(model, genre_matrix):
    model.eval()
    with torch.no_grad():
        genre_tensor = torch.tensor(genre_matrix, dtype=torch.float32)
        embeddings = model.genre_embedding(genre_tensor)
        normalized = torch.nn.functional.normalize(embeddings, dim=1)
    return normalized.cpu().numpy()


def recommend_by_genre(
    target_movie_id,
    movies_df,
    movie_id_to_idx,
    genre_embeddings,
    movie_id_to_imdb=None,
    title_tokens=None,
    movie_id_to_avg_rating=None,
    franchise_keys=None,
    years=None,
    avg_rating_range=None,
    min_year=None,
    max_year=None,
    min_rating=None,
    max_rating=None,
    genre_weight=0.6,
    title_weight=0.2,
    franchise_weight=0.2,
    year_weight=0.00,
    rating_weight=0.00,
    top_k=10,
):
    if target_movie_id not in movie_id_to_idx:
        raise ValueError("Movie id not found in mapping.")

    target_idx = movie_id_to_idx[target_movie_id]
    target_vec = genre_embeddings[target_idx]
    genre_sims = np.maximum(genre_embeddings @ target_vec, 0.0)
    genre_sims[target_idx] = -np.inf  # avoid recommending the same movie

    title_sims = np.zeros_like(genre_sims)
    if title_tokens:
        target_tokens = title_tokens[target_idx] if target_idx < len(title_tokens) else set()
        if target_tokens:
            for idx, tokens in enumerate(title_tokens):
                if idx == target_idx:
                    continue
                title_sims[idx] = _title_similarity(target_tokens, tokens)

    franchise_sims = np.zeros_like(genre_sims)
    if franchise_keys:
        target_franchise = franchise_keys[target_idx] if target_idx < len(franchise_keys) else ""
        for idx, key in enumerate(franchise_keys):
            if idx == target_idx:
                continue
            franchise_sims[idx] = 1.0 if target_franchise and key == target_franchise else 0.0

    year_sims = np.zeros_like(genre_sims)
    if years:
        target_year = years[target_idx] if target_idx < len(years) else None
        for idx, year in enumerate(years):
            if idx == target_idx:
                continue
            year_sims[idx] = _year_similarity(target_year, year)

    rating_sims = np.zeros_like(genre_sims)
    if movie_id_to_avg_rating:
        for idx, row in movies_df.iterrows():
            rating = movie_id_to_avg_rating.get(int(row.movieId))
            if rating is not None:
                rating_sims[idx] = _normalize_rating(rating, avg_rating_range)

    mask = np.ones_like(genre_sims, dtype=bool)
    if years and (min_year is not None or max_year is not None):
        for idx, year in enumerate(years):
            if year is None:
                continue
            if min_year is not None and year < min_year:
                mask[idx] = False
            if max_year is not None and year > max_year:
                mask[idx] = False
    if movie_id_to_avg_rating and (min_rating is not None or max_rating is not None):
        for idx, row in movies_df.iterrows():
            rating = movie_id_to_avg_rating.get(int(row.movieId))
            if rating is None:
                mask[idx] = False
                continue
            if min_rating is not None and rating < min_rating:
                mask[idx] = False
            if max_rating is not None and rating > max_rating:
                mask[idx] = False
    mask[target_idx] = False

    weights_sum = genre_weight + title_weight + franchise_weight + year_weight + rating_weight
    if weights_sum <= 0:
        genre_w = 1.0
        title_w = franchise_w = year_w = rating_w = 0.0
    else:
        genre_w = genre_weight / weights_sum
        title_w = title_weight / weights_sum
        franchise_w = franchise_weight / weights_sum
        year_w = year_weight / weights_sum
        rating_w = rating_weight / weights_sum

    combined_sims = (
        genre_w * genre_sims
        + title_w * title_sims
        + franchise_w * franchise_sims
        + year_w * year_sims
        + rating_w * rating_sims
    )
    combined_sims[target_idx] = -np.inf
    combined_sims = np.where(mask, combined_sims, -np.inf)
    top_indices = np.argsort(combined_sims)[::-1][:top_k]

    recommendations = []
    for idx in top_indices:
        row = movies_df.iloc[idx]
        imdb_id = None
        imdb_url = None
        if movie_id_to_imdb:
            imdb_id = movie_id_to_imdb.get(int(row.movieId))
            if imdb_id:
                imdb_url = f"https://www.imdb.com/title/tt{imdb_id}/"
        avg_rating = None
        if movie_id_to_avg_rating:
            avg_rating = movie_id_to_avg_rating.get(int(row.movieId))
        recommendations.append(
            {
                "movieId": int(row.movieId),
                "imdbId": imdb_id,
                "imdbUrl": imdb_url,
                "title": row.title,
                "genres": row.genres,
                "avgRating": float(avg_rating) if avg_rating is not None else None,
                "score": float(combined_sims[idx]),
            }
        )

    return recommendations


def _title_similarity(a, b):
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    return overlap / max(len(a), len(b))


def _year_similarity(a, b):
    if a is None or b is None:
        return 0.0
    gap = abs(a - b)
    return max(0.0, 1.0 - gap / 15.0)


def _normalize_rating(rating, rating_range):
    if not rating_range:
        return 0.0
    r_min, r_max = rating_range
    if r_max <= r_min:
        return 0.0
    return (rating - r_min) / (r_max - r_min)
