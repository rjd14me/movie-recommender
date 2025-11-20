from src.data_preparation import load_raw_data, attach_movie_stats
from src.models.deep_model import DeepRecommender, DeepRecommenderConfig


def test_deep_model_small_run():
  ratings, movies = load_raw_data()
  movies_with_stats = attach_movie_stats(ratings, movies)

  cfg = DeepRecommenderConfig(n_factors=8, n_epochs=1, batch_size=2048)
  model = DeepRecommender(config=cfg)
  model.fit(ratings.sample(min(len(ratings), 5000), random_state=0), movies_with_stats)

  some_movie_id = int(movies_with_stats["movieId"].iloc[0])
  recs = model.similar_movies(some_movie_id, n=5)
  assert not recs.empty
