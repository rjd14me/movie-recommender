# src/data_preparation.py
from pathlib import Path
import pandas as pd


def load_raw_data(data_dir: str = "data"):
    """
    Load ratings and movies CSVs from the data directory.
    Expects MovieLens-style 'ratings.csv' and 'movies.csv'.
    """
    data_path = Path(data_dir)

    ratings_path = data_path / "ratings.csv"
    movies_path = data_path / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"Could not find ratings.csv or movies.csv in {data_path.resolve()}"
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    return ratings, movies


def prepare_merged_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Convenience function: load and merge ratings + movies into one DataFrame.
    """
    ratings, movies = load_raw_data(data_dir=data_dir)
    merged = ratings.merge(movies, on="movieId", how="left")
    return merged


if __name__ == "__main__":
    # Simple sanity check when you run:
    # python -m src.data_preparation
    merged = prepare_merged_data()
    print(merged.head())
    print(f"Total rows: {len(merged)}")
