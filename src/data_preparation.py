import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_dir="data"):
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    movies = pd.read_csv(movies_path)
    movies["title"] = movies["title"].apply(normalize_title)
    movies["title_tokens"] = movies["title"].apply(tokenize_title)
    movies["year"] = movies["title"].apply(extract_year)
    movies["franchise_key"] = movies["title"].apply(build_franchise_key)
    movies["search_key"] = movies["title"].apply(build_search_key)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def load_imdb_links(data_dir="data"):
    links_path = os.path.join(data_dir, "links.csv")
    if not os.path.exists(links_path):
        return {}

    links_df = pd.read_csv(
        links_path,
        dtype={"movieId": int, "imdbId": str},
        usecols=["movieId", "imdbId"],
        keep_default_na=False,
    )

    mapping = {}
    for _, row in links_df.iterrows():
        imdb_id = str(row["imdbId"]).strip()
        if imdb_id:
            padded = imdb_id.zfill(7)
            mapping[int(row["movieId"])] = padded

    return mapping


ARTICLE_SUFFIX_RE = re.compile(r"^(?P<body>.+),\s*(?P<article>The|An|A)\s*(?P<year>\(\d{4}.*\))?$")
TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
ROMAN_NUMERAL_RE = re.compile(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", re.IGNORECASE)


def normalize_title(title):
    if not isinstance(title, str):
        return title

    match = ARTICLE_SUFFIX_RE.match(title.strip())
    if not match:
        return title

    body = match.group("body").strip()
    article = match.group("article")
    year = match.group("year") or ""
    spacer = " " if year else ""
    return f"{article} {body}{spacer}{year}".strip()


def tokenize_title(title):
    if not isinstance(title, str):
        return set()

    cleaned = title.strip()
    year_match = re.search(r"\(\d{4}.*\)$", cleaned)
    if year_match:
        cleaned = cleaned[: year_match.start()].strip()

    raw_tokens = TITLE_TOKEN_RE.findall(cleaned)
    lower_tokens = []
    for token in raw_tokens:
        lower_tokens.append(token.lower())

    stop_words = {"a", "an", "the"}
    unique_tokens = set()
    for token in lower_tokens:
        if token and token not in stop_words:
            unique_tokens.add(token)

    return unique_tokens


def extract_year(title):
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})", title)
    if not match:
        return None

    year_text = match.group(1)
    if not year_text.isdigit():
        return None

    return int(year_text)


def build_franchise_key(title):
    if not isinstance(title, str):
        return ""

    cleaned = re.sub(r"\(\d{4}.*\)", "", title).strip()
    cleaned = ROMAN_NUMERAL_RE.sub("", cleaned)
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def build_search_key(title):
    if not isinstance(title, str):
        return ""

    cleaned = re.sub(r"\(\d{4}.*\)", "", title).strip()
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned).lower()
    return cleaned


def build_genre_mapping(movies):
    genre_set = set()
    for cell in movies["genres"].fillna(""):
        for genre in cell.split("|"):
            if genre and genre != "(no genres listed)":
                genre_set.add(genre)

    sorted_genres = sorted(genre_set)

    mapping = {}
    for idx, genre in enumerate(sorted_genres):
        mapping[genre] = idx

    return mapping


def encode_genres(movies, genre_to_idx):
    num_movies = len(movies)
    num_genres = len(genre_to_idx)
    genre_matrix = np.zeros((num_movies, num_genres), dtype=np.float32)

    for row_idx, cell in enumerate(movies["genres"].fillna("")):
        for genre in cell.split("|"):
            genre_idx = genre_to_idx.get(genre)
            if genre_idx is not None:
                genre_matrix[row_idx, genre_idx] = 1.0

    return genre_matrix


def build_id_mappings(movies, filtered_ratings):
    movie_id_to_idx = {}
    for idx, movie_id in enumerate(movies["movieId"].tolist()):
        movie_id_to_idx[int(movie_id)] = idx

    user_ids = sorted(filtered_ratings["userId"].unique())
    user_id_to_idx = {}
    for idx, user_id in enumerate(user_ids):
        user_id_to_idx[int(user_id)] = idx

    return movie_id_to_idx, user_id_to_idx


def filter_ratings(ratings, min_ratings=10):
    counts = ratings["userId"].value_counts()
    keep_users = counts[counts >= min_ratings].index
    return ratings[ratings["userId"].isin(keep_users)].reset_index(drop=True)


def prepare_datasets(
    data_dir="data", test_size=0.1, min_ratings=10
):
    movies_df, ratings_df = load_data(data_dir)
    ratings_df = filter_ratings(ratings_df, min_ratings=min_ratings)
    genre_to_idx = build_genre_mapping(movies_df)
    genre_matrix = encode_genres(movies_df, genre_to_idx)
    movie_id_to_idx, user_id_to_idx = build_id_mappings(movies_df, ratings_df)

    ratings_df["movie_idx"] = ratings_df["movieId"].map(movie_id_to_idx)
    ratings_df["user_idx"] = ratings_df["userId"].map(user_id_to_idx)

    user_array = ratings_df["user_idx"].to_numpy()
    movie_array = ratings_df["movie_idx"].to_numpy()
    rating_array = ratings_df["rating"].astype(np.float32).to_numpy()

    X_train_user, X_val_user, X_train_movie, X_val_movie, y_train, y_val = train_test_split(
        user_array, movie_array, rating_array, test_size=test_size, random_state=42, shuffle=True
    )

    splits = (
        X_train_user,
        X_val_user,
        X_train_movie,
        X_val_movie,
        y_train,
        y_val,
    )

    return (
        movies_df,
        ratings_df,
        genre_to_idx,
        genre_matrix,
        movie_id_to_idx,
        user_id_to_idx,
        splits,
    )
