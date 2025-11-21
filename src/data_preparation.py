from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fix_title(title: str) -> str:
    if "(" not in title:
        return title

    try:
        name_part, year_part = title.rsplit("(", 1)
        year = year_part.strip(") ").strip()
        name_part = name_part.strip()

        pieces = name_part.split(", ")
        if len(pieces) > 1 and pieces[-1] in {"The", "A", "An"}:
            article = pieces[-1]
            rest = ", ".join(pieces[:-1])
            fixed = f"{article} {rest} ({year})"
            return fixed

        return f"{name_part} ({year})"
    except Exception:
        return title


def load_movies() -> pd.DataFrame:
    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found at {movies_path.resolve()}")

    movies = pd.read_csv(movies_path)
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["title"] = movies["title"].apply(fix_title)
    return movies


def load_ratings() -> pd.DataFrame:
    ratings_path = DATA_DIR / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found at {ratings_path.resolve()}")

    ratings = pd.read_csv(ratings_path)
    return ratings


def load_links() -> pd.DataFrame:
    links_path = DATA_DIR / "links.csv"
    if not links_path.exists():
        raise FileNotFoundError(f"links.csv not found at {links_path.resolve()}")

    links = pd.read_csv(links_path)

    if "imdbId" not in links.columns:
        raise ValueError("links.csv is missing required column 'imdbId'.")
    links = links.dropna(subset=["imdbId"])

    links["imdbId"] = links["imdbId"].astype("int64").astype(str).str.zfill(7)
    links["imdb_url"] = "https://www.imdb.com/title/tt" + links["imdbId"]

    return links[["movieId", "imdb_url"]]


def load_raw_data():
    ratings = load_ratings()
    movies = load_movies()

    try:
        links = load_links()
    except FileNotFoundError:
        links = None

    if links is not None:
        movies = movies.merge(links, on="movieId", how="left")

    return ratings, movies


def attach_movie_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Adds rating_mean and rating_count columns to movies."""
    grouped = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"mean": "rating_mean", "count": "rating_count"}
    )
    movies_with_stats = movies.merge(grouped, on="movieId", how="left")
    return movies_with_stats


def load_movies_with_stats():
    ratings, movies = load_raw_data()
    movies_with_stats = attach_movie_stats(ratings, movies)
    return ratings, movies_with_stats
