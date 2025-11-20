# Movie → Movie Recommender (Deep Learning + Web UI)

This project is a small-but-complete movie recommender system designed to
showcase:

- Python & project structure
- FastAPI backend
- Deep-learning embeddings for recommendations (PyTorch)
- Modern frontend (HTML/CSS/JS) with live search suggestions

It uses the [MovieLens 100k](https://grouplens.org/datasets/movielens/) style
`movies.csv` and `ratings.csv` files.

---

## Project Structure

```text
movie-recommender/
  data/
    movies.csv
    ratings.csv
  src/
    data_preparation.py
    api.py
    evaluation.py
    models/
      deep_model.py
  templates/
    index.html
  static/
    style.css
  requirements.txt
  README.md
