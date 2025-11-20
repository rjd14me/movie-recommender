from pathlib import Path

from src.data_preparation import load_raw_data, attach_movie_stats
from src.models.deep_model import DeepRecommender, DeepRecommenderConfig


def main():
    ratings, movies = load_raw_data()
    movies_with_stats = attach_movie_stats(ratings, movies)

    cfg = DeepRecommenderConfig(
        n_factors=64,
        n_epochs=5,
        batch_size=4096,
        lr=5e-3,
        weight_decay=1e-6,
        min_rating_count=5,
    )

    model = DeepRecommender(config=cfg)
    model.fit(ratings, movies_with_stats)

    out_path = Path("data") / "deep_model.pt"
    model.save(out_path)
    print(f"Saved model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
