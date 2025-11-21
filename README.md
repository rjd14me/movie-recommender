# Movie Recommender (v1.1)

This is the first Version of a small movie recommender system built with Python and FastAPI.

- Uses data from the MovieLens set  (`movies.csv` and `ratings.csv`)
- Learns movie patterns
- Displays a simple web UI where you type a movie title and get similar movies, it allows users to get movies with a similar dataset(genre)
- Includes a live search dropdown that updates as you type, allowing users to quickly select specific movies.

The goal is to build basic experience with:

- Python project structure
- FastAPI
- Simple deep learning (PyTorch)
- Frontend (HTML/CSS/JS)

identified issues with this version
- extremely inaccurate results sometimes, e.g. Good Will Hunting will come up with The Lord Of the Rings with the highest match rating despite having different genres.
- search bar does not allow users room for error.
- lack of features, such as ability to filter reccomendations by year of release, or genres.
- the ai does not have enough data to make very accurate decisions.
---

## Project layout

```text
movie-recommender/
  data/                 # movies.csv, ratings.csv (not tracked in Git)
  src/
    api.py
    data_preparation.py
    evaluation.py
    models/
      deep_model.py
  templates/
    index.html
  static/
    style.css
  tests/
    test_deep_model.py
  requirements.txt
  README.md
  .gitignore
