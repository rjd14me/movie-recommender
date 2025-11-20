# src/evaluation.py
import pandas as pd

from src.data_preparation import load_raw_data
from src.models.baseline_cf import PopularityRecommender



def train_test_split_last_n(ratings: pd.DataFrame, n: int = 1):
    """
    For each user, hold out their last n ratings as test, rest as train.
    Uses the timestamp column to determine "last".
    """
    # Sort by user, then by timestamp
    ratings_sorted = ratings.sort_values(["userId", "timestamp"])

    def split_user(group):
        if len(group) <= n:
            return group.iloc[:0], group  # all in test
        return group.iloc[:-n], group.iloc[-n:]

    train_parts = []
    test_parts = []

    for _, user_group in ratings_sorted.groupby("userId"):
        train_g, test_g = split_user(user_group)
        train_parts.append(train_g)
        test_parts.append(test_g)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def hit_rate_at_k(model: PopularityRecommender, test_ratings: pd.DataFrame, k: int = 10):
    """
    Measure how often the held-out item appears in the top-k recommendations.
    """
    hits = 0
    total = 0

    for user_id, user_group in test_ratings.groupby("userId"):
        true_movie_ids = set(user_group["movieId"].unique())
        recs = model.recommend_for_user(user_id=user_id, n=k, exclude_seen=True)

        if recs.empty:
            continue

        rec_movie_ids = set(recs["movieId"].tolist())

        if true_movie_ids & rec_movie_ids:
            hits += 1

        total += 1

    if total == 0:
        return 0.0

    return hits / total


if __name__ == "__main__":
    # Example usage:
    ratings, movies = load_raw_data()

    train_ratings, test_ratings = train_test_split_last_n(ratings, n=1)

    model = PopularityRecommender(min_ratings=50)
    model.fit(train_ratings, movies)

    hr10 = hit_rate_at_k(model, test_ratings, k=10)
    print(f"Hit Rate@10: {hr10:.3f}")
