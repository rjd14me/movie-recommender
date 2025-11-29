import difflib
import os
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.data_preparation import load_imdb_links, prepare_datasets
from src.models.deep_model import compute_genre_embeddings, recommend_by_genre, train_genre_model

app = FastAPI(title="Movie Recommender", version="1.4")
app.mount("/static", StaticFiles(directory=os.path.join("static")), name="static")
templates = Jinja2Templates(directory=os.path.join("templates"))

(
    MOVIES_DF,
    RATINGS_DF,
    GENRE_TO_IDX,
    GENRE_MATRIX,
    MOVIE_ID_TO_IDX,
    USER_ID_TO_IDX,
    SPLITS,
) = prepare_datasets()

MODEL, METRICS = train_genre_model(
    train_users=SPLITS[0],
    val_users=SPLITS[1],
    train_movies=SPLITS[2],
    val_movies=SPLITS[3],
    train_ratings=SPLITS[4],
    val_ratings=SPLITS[5],
    genre_matrix=GENRE_MATRIX,
    num_users=len(USER_ID_TO_IDX),
    num_genres=len(GENRE_TO_IDX),
    epochs=3,
)

GENRE_EMBEDDINGS = compute_genre_embeddings(MODEL, GENRE_MATRIX)
MOVIE_ID_TO_IMDB = load_imdb_links()
MOVIE_TITLE_TOKENS = (
    MOVIES_DF["title_tokens"].tolist() if "title_tokens" in MOVIES_DF.columns else [set()] * len(MOVIES_DF)
)
MOVIE_FRANCHISE_KEYS = MOVIES_DF["franchise_key"].fillna("").tolist() if "franchise_key" in MOVIES_DF.columns else []
MOVIE_YEARS = MOVIES_DF["year"].tolist() if "year" in MOVIES_DF.columns else []
MOVIE_AVG_RATINGS = RATINGS_DF.groupby("movieId")["rating"].mean().to_dict()
if MOVIE_AVG_RATINGS:
    AVG_RATING_RANGE = (min(MOVIE_AVG_RATINGS.values()), max(MOVIE_AVG_RATINGS.values()))
else:
    AVG_RATING_RANGE = None


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/search")
async def search_movies(q: str = Query(..., description="Partial movie title to search")):
    lowered = q.strip().lower()
    if not lowered:
        return []

    query_key = re.sub(r"[^a-z0-9\s]", " ", lowered)
    query_key = re.sub(r"\s+", "", query_key)
    if len(query_key) < 2:
        return []

    scored = []
    min_ratio = 0.65 if len(query_key) <= 5 else 0.55
    for idx, row in MOVIES_DF.iterrows():
        search_key = row.search_key
        if not isinstance(search_key, str) or not search_key:
            continue

        ratio = difflib.SequenceMatcher(None, query_key, search_key).ratio()
        if query_key and query_key in search_key:
            ratio += 0.25
            if search_key.startswith(query_key):
                ratio += 0.1

        if ratio >= min_ratio:
            scored.append((ratio, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:12]

    return [
        {"movieId": int(MOVIES_DF.iloc[idx].movieId), "title": MOVIES_DF.iloc[idx].title, "genres": MOVIES_DF.iloc[idx].genres}
        for _, idx in top
    ]


@app.get("/api/random")
async def random_movie():
    sample = MOVIES_DF.sample(1).iloc[0]
    return {"movieId": int(sample.movieId), "title": sample.title, "genres": sample.genres}


@app.get("/api/recommend")
async def recommend(
    movie_id: int = Query(..., description="Movie id to use as the anchor"),
    limit: int = Query(10, ge=1, le=25, description="Number of recommendations to return"),
    min_year: int | None = Query(None, ge=1800, le=2100, description="Earliest release year to include"),
    max_year: int | None = Query(None, ge=1800, le=2100, description="Latest release year to include"),
    min_rating: float | None = Query(None, ge=0.0, le=5.0, description="Minimum average rating"),
    max_rating: float | None = Query(None, ge=0.0, le=5.0, description="Maximum average rating"),
):
    if movie_id not in MOVIE_ID_TO_IDX:
        raise HTTPException(status_code=404, detail="Movie id not found.")

    try:
        recommendations = recommend_by_genre(
            target_movie_id=movie_id,
            movies_df=MOVIES_DF,
            movie_id_to_idx=MOVIE_ID_TO_IDX,
            genre_embeddings=GENRE_EMBEDDINGS,
            movie_id_to_imdb=MOVIE_ID_TO_IMDB,
            title_tokens=MOVIE_TITLE_TOKENS,
            movie_id_to_avg_rating=MOVIE_AVG_RATINGS,
            franchise_keys=MOVIE_FRANCHISE_KEYS,
            years=MOVIE_YEARS,
            avg_rating_range=AVG_RATING_RANGE,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            max_rating=max_rating,
            genre_weight=0.6,
            title_weight=0.2,
            franchise_weight=0.2,
            year_weight=0.0,
            rating_weight=0.00,
            top_k=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "movieId": movie_id,
        "recommendations": recommendations,
        "metrics": METRICS,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "val_mse": METRICS.get("val_mse")}


if __name__ == "__main__":
    import os
    import socket
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port_raw = os.getenv("PORT", "8000")
    reload_enabled = os.getenv("RELOAD", "false").lower() not in {"0", "false", "no"}

    def pick_free_port(bind_host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((bind_host, 0))
            return s.getsockname()[1]

    try:
        port = 0 if port_raw in {"0", "auto", ""} else int(port_raw)
    except ValueError:
        port = 0

    if port == 0:
        port = pick_free_port(host)

    try:
        uvicorn.run("src.api:app", host=host, port=port, reload=reload_enabled)
    except PermissionError:
        safe_host = "127.0.0.1"
        fallback_port_raw = os.getenv("PORT_FALLBACK", "0")
        try:
            fallback_port = int(fallback_port_raw)
        except ValueError:
            fallback_port = 0
        if fallback_port == 0:
            fallback_port = pick_free_port(safe_host)
        print(f"Permission denied on {host}:{port}, retrying on {safe_host}:{fallback_port}")
        uvicorn.run("src.api:app", host=safe_host, port=fallback_port, reload=False)
