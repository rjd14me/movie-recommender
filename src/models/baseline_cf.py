# src/models/baseline_cf.py
import pandas as pd


class PopularityRecommender:
    """
    Very simple baseline recommender:
    - Computes average rating and rating count per movie
    - Recommends the most popular, well-rated movies
    """

    def __init__(self, min_ratings: int = 50):
        self.min_ratings = min_ratings
        self.movie_rankings = None
        self.ratings = None
        self.movies = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        ratings: DataFrame with columns [userId, movieId, rating, timestamp]
        movies:  DataFrame with columns [movieId, title, genres]
        """
        self.ratings = ratings.copy()
        self.movies = movies.copy()

        movie_stats = (
            ratings.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                rating_mean=("rating", "mean"),
            )
            .reset_index()
        )

        movie_stats = movie_stats.merge(movies, on="movieId", how="left")

        # Filter to movies with enough ratings
        filtered = movie_stats[movie_stats["rating_count"] >= self.min_ratings]

        # Sort best first
        filtered = filtered.sort_values(
            ["rating_mean", "rating_count"], ascending=False
        )

        self.movie_rankings = filtered

    def recommend_for_user(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> pd.DataFrame:
        """
        Return a DataFrame of top-N recommendations for a given user.
        """
        if self.movie_rankings is None:
            raise RuntimeError("You must call .fit() before .recommend_for_user().")

        recommendations = self.movie_rankings.copy()

        if exclude_seen:
            user_ratings = self.ratings[self.ratings["userId"] == user_id]
            seen_movie_ids = set(user_ratings["movieId"].unique())
            recommendations = recommendations[
                ~recommendations["movieId"].isin(seen_movie_ids)
            ]

        return recommendations.head(n).reset_index(drop=True)
