from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data_preparation import load_raw_data, attach_movie_stats
from src.models.deep_model import DeepRecommender, DeepRecommenderConfig

app = FastAPI(
    title="Movie Recommender",
    description="Simple movie-to-movie recommender with a web UI.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

deep_recommender: Optional[DeepRecommender] = None


@app.on_event("startup")
def startup_event():
    global deep_recommender

    ratings, movies = load_raw_data()
    movies_with_stats = attach_movie_stats(ratings, movies)

    model_path = Path("data") / "deep_model.pt"
    if model_path.exists():
        print(f"[startup] Loading model from {model_path.resolve()}")
        deep_recommender = DeepRecommender.load_from_file(model_path, movies_with_stats)
        print("[startup] Model loaded.")
    else:
        print("[startup] No saved model, training a small one now.")
        cfg = DeepRecommenderConfig(
            n_factors=32,
            n_epochs=3,
            batch_size=4096,
            lr=5e-3,
            weight_decay=1e-6,
            min_rating_count=5,
        )
        model = DeepRecommender(config=cfg)
        model.fit(ratings, movies_with_stats)
        model.save(model_path)
        deep_recommender = model
        print(f"[startup] Model trained and saved to {model_path.resolve()}")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/search_titles")
def search_titles(q: str, limit: int = 5):
    global deep_recommender

    if deep_recommender is None or deep_recommender.movies is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    query = q.strip()
    if not query or len(query) < 2:
        return []

    movies = deep_recommender.movies
    titles = movies["title"].astype(str)

    mask = titles.str.contains(query, case=False, na=False)
    matches = movies[mask].copy()

    if "rating_count" in matches.columns:
        matches = matches.sort_values("rating_count", ascending=False)
    else:
        matches = matches.sort_values("title")

    cols = ["movieId", "title", "genres"]
    if "rating_mean" in matches.columns:
        cols.append("rating_mean")
    if "rating_count" in matches.columns:
        cols.append("rating_count")

    return matches[cols].head(limit).to_dict(orient="records")


@app.get("/api/similar_by_title")
def get_similar_by_title(title: str, n: int = 10):
    global deep_recommender

    if deep_recommender is None or deep_recommender.movies is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    movies = deep_recommender.movies
    titles = movies["title"].astype(str)
    query = title.strip().lower()

    if not query:
        raise HTTPException(status_code=400, detail="Title must not be empty.")

    titles_lower = titles.str.lower()
    exact_mask = titles_lower == query
    exact_matches = movies[exact_mask]

    if not exact_matches.empty:
        chosen = exact_matches.iloc[0]
    else:
        mask = titles_lower.str.contains(query)
        matches = movies[mask]
        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail="No movie found matching that title.",
            )
        chosen = matches.iloc[0]

    movie_id = int(chosen["movieId"])

    try:
        recs = deep_recommender.similar_movies(movie_id=movie_id, n=n)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Movie not present in the trained model.",
        )

    cols = ["movieId", "title", "genres", "similarity"]
    if "rating_mean" in recs.columns:
        cols.append("rating_mean")
    if "rating_count" in recs.columns:
        cols.append("rating_count")

    return {
        "input_movie": {
            "movieId": movie_id,
            "title": chosen.get("title"),
            "genres": chosen.get("genres"),
            "rating_mean": chosen.get("rating_mean"),
            "rating_count": chosen.get("rating_count"),
        },
        "recommendations": recs[cols].to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
