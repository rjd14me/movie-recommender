# src/api.py
from fastapi import FastAPI, HTTPException

from src.data_preparation import load_raw_data
from src.models.baseline_cf import PopularityRecommender

app = FastAPI(
    title="Movie Recommender API (v1)",
    description="Simple popularity-based recommender using MovieLens data.",
    version="0.1.0",
)

recommender: PopularityRecommender | None = None


@app.on_event("startup")
def startup_event():
    global recommender

    ratings, movies = load_raw_data()
    model = PopularityRecommender(min_ratings=50)
    model.fit(ratings, movies)
    recommender = model
    print("Recommender model loaded and ready!")


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, n: int = 10):
    global recommender

    if recommender is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    recs = recommender.recommend_for_user(user_id=user_id, n=n, exclude_seen=True)

    if recs.empty:
        raise HTTPException(
            status_code=404,
            detail="No recommendations available for this user.",
        )

    # Return as a list of dicts (JSON)
    return recs[["movieId", "title", "rating_mean", "rating_count"]].to_dict(
        orient="records"
    )


# Allow `python -m src.api` to run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
