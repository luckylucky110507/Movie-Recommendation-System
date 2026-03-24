from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_TITLE = "Movie Recommendation System"
APP_SUBTITLE = "Discover similar Hollywood and Bollywood movies with a content-based machine learning model."
DATA_DIR = Path("data")
DATA_FILES = [
    DATA_DIR / "movies.csv",
    DATA_DIR / "hollywood_movies.csv",
    DATA_DIR / "bollywood_movies.csv",
]
REQUIRED_COLUMNS = [
    "title",
    "industry",
    "genre",
    "year",
    "director",
    "cast",
    "rating",
    "description",
]

FALLBACK_MOVIES = [
    {
        "title": "The Shawshank Redemption",
        "industry": "Hollywood",
        "genre": "Drama",
        "year": 1994,
        "director": "Frank Darabont",
        "cast": "Tim Robbins, Morgan Freeman",
        "rating": 9.3,
        "description": "Two imprisoned men form a lasting friendship while they dream of freedom.",
    },
    {
        "title": "The Dark Knight",
        "industry": "Hollywood",
        "genre": "Action Crime Drama",
        "year": 2008,
        "director": "Christopher Nolan",
        "cast": "Christian Bale, Heath Ledger",
        "rating": 9.0,
        "description": "Batman faces the Joker, a chaotic criminal who pushes Gotham to the edge.",
    },
    {
        "title": "Inception",
        "industry": "Hollywood",
        "genre": "Sci-Fi Action Thriller",
        "year": 2010,
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt",
        "rating": 8.8,
        "description": "A thief enters dreams to steal secrets and is offered a final chance at redemption.",
    },
    {
        "title": "3 Idiots",
        "industry": "Bollywood",
        "genre": "Comedy Drama",
        "year": 2009,
        "director": "Rajkumar Hirani",
        "cast": "Aamir Khan, R Madhavan, Sharman Joshi",
        "rating": 8.4,
        "description": "Three engineering students navigate friendship, pressure, and ambition.",
    },
    {
        "title": "Dangal",
        "industry": "Bollywood",
        "genre": "Biography Drama Sport",
        "year": 2016,
        "director": "Nitesh Tiwari",
        "cast": "Aamir Khan, Fatima Sana Shaikh, Sanya Malhotra",
        "rating": 8.3,
        "description": "A former wrestler trains his daughters to become world-class champions.",
    },
    {
        "title": "Zindagi Na Milegi Dobara",
        "industry": "Bollywood",
        "genre": "Adventure Comedy Drama",
        "year": 2011,
        "director": "Zoya Akhtar",
        "cast": "Hrithik Roshan, Farhan Akhtar, Abhay Deol",
        "rating": 8.2,
        "description": "Three friends rediscover themselves on a road trip across Spain.",
    },
]


def normalize_industry(value):
    clean_value = str(value).strip().lower()
    mapping = {
        "hollywood": "Hollywood",
        "bollywood": "Bollywood",
    }
    return mapping.get(clean_value, str(value).strip().title())


def sanitize_movie_dataframe(movie_df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in movie_df.columns]
    if missing_columns:
        raise ValueError("Missing required columns in dataset: " + ", ".join(missing_columns))

    movie_df = movie_df[REQUIRED_COLUMNS].copy()
    for column in ["title", "industry", "genre", "director", "cast", "description"]:
        movie_df[column] = movie_df[column].fillna("").astype(str).str.strip()

    movie_df["industry"] = movie_df["industry"].apply(normalize_industry)
    movie_df["year"] = pd.to_numeric(movie_df["year"], errors="coerce")
    movie_df["rating"] = pd.to_numeric(movie_df["rating"], errors="coerce")
    movie_df = movie_df.dropna(subset=["title", "industry", "year"])
    movie_df["year"] = movie_df["year"].astype(int)
    movie_df["rating"] = movie_df["rating"].fillna(0.0).clip(lower=0, upper=10).round(1)
    movie_df = movie_df[movie_df["title"] != ""]
    movie_df = movie_df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    movie_df["features"] = (
        movie_df["industry"]
        + " "
        + movie_df["genre"]
        + " "
        + movie_df["director"]
        + " "
        + movie_df["cast"]
        + " "
        + movie_df["description"]
    )
    return movie_df


@st.cache_data
def load_movies():
    available_files = [file_path for file_path in DATA_FILES if file_path.exists()]

    if available_files:
        dataframes = [pd.read_csv(file_path) for file_path in available_files]
        movie_df = pd.concat(dataframes, ignore_index=True)
    else:
        movie_df = pd.DataFrame(FALLBACK_MOVIES)

    return sanitize_movie_dataframe(movie_df), available_files


@st.cache_data
def parse_uploaded_dataset(uploaded_file):
    uploaded_df = pd.read_csv(uploaded_file)
    return sanitize_movie_dataframe(uploaded_df)


@st.cache_resource
def train_recommender(movie_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(movie_df["features"])
    similarity_matrix = cosine_similarity(feature_matrix)
    title_to_index = pd.Series(movie_df.index, index=movie_df["title"].str.lower()).to_dict()
    return similarity_matrix, title_to_index


def recommend_movies(movie_df, similarity_matrix, title_to_index, title, industry_filter, genre_filter, min_year, top_n):
    selected_index = title_to_index[title.strip().lower()]
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))
    similarity_scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for movie_index, score in similarity_scores:
        if movie_index == selected_index:
            continue

        movie = movie_df.iloc[movie_index]
        if industry_filter != "All" and movie["industry"] != industry_filter:
            continue
        if genre_filter != "All" and genre_filter.lower() not in movie["genre"].lower():
            continue
        if movie["year"] < min_year:
            continue

        recommendations.append(
            {
                "Title": movie["title"],
                "Industry": movie["industry"],
                "Genre": movie["genre"],
                "Year": int(movie["year"]),
                "Director": movie["director"],
                "IMDb Rating": float(movie["rating"]),
                "Match %": round(float(score) * 100, 2),
            }
        )

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


def filter_movie_options(movie_df, industry_filter, search_term):
    filtered_df = movie_df.copy()
    if industry_filter != "All":
        filtered_df = filtered_df[filtered_df["industry"] == industry_filter]
    if search_term.strip():
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(search_term.strip(), case=False, regex=False)
        ]
    return filtered_df


def build_download_template():
    return pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False).encode("utf-8")


def top_rated_movies(movie_df, industry_filter):
    filtered_df = movie_df if industry_filter == "All" else movie_df[movie_df["industry"] == industry_filter]
    return (
        filtered_df.sort_values(by=["rating", "year"], ascending=[False, False])
        .loc[:, ["title", "industry", "genre", "year", "rating"]]
        .head(10)
        .rename(
            columns={
                "title": "Title",
                "industry": "Industry",
                "genre": "Genre",
                "year": "Year",
                "rating": "IMDb Rating",
            }
        )
    )


st.set_page_config(page_title=APP_TITLE, page_icon=":movie_camera:", layout="wide")

try:
    base_movies_df, available_files = load_movies()
except Exception as error:
    st.error(f"Dataset error: {error}")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.download_button(
        "Download CSV Template",
        data=build_download_template(),
        file_name="movie_dataset_template.csv",
        mime="text/csv",
    )

if uploaded_file is not None:
    try:
        movies_df = parse_uploaded_dataset(uploaded_file)
        dataset_label = f"uploaded file: {uploaded_file.name}"
    except Exception as error:
        st.error(f"Uploaded dataset error: {error}")
        st.stop()
else:
    movies_df = base_movies_df
    dataset_label = ", ".join(file_path.name for file_path in available_files) if available_files else "fallback sample"

similarity_matrix, title_to_index = train_recommender(movies_df)

industry_choices = ["All"] + sorted(movies_df["industry"].dropna().unique().tolist())
genre_choices = ["All"] + sorted(
    {
        genre
        for value in movies_df["genre"]
        for genre in value.split()
        if genre
    }
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Filters")
    selected_industry = st.selectbox("Industry filter", industry_choices)
    search_term = st.text_input("Search movie title", placeholder="Type a movie name")
    movie_options_df = filter_movie_options(movies_df, selected_industry, search_term)

    if movie_options_df.empty:
        st.warning("No movies match the current search or industry filter.")
        st.stop()

    selected_movie = st.selectbox("Choose a movie", sorted(movie_options_df["title"].tolist()))
    selected_genre = st.selectbox("Genre filter", genre_choices)
    selected_year = st.slider(
        "Minimum release year",
        min_value=int(movies_df["year"].min()),
        max_value=int(movies_df["year"].max()),
        value=int(movies_df["year"].min()),
    )
    top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=8)

selected_movie_row = movies_df.loc[movies_df["title"] == selected_movie].iloc[0]

metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Movies Loaded", len(movies_df))
metric_two.metric("Industries", movies_df["industry"].nunique())
metric_three.metric("Genres", len(genre_choices) - 1)

overview_tab, recommendations_tab, dataset_tab = st.tabs(
    ["Overview", "Recommendations", "Dataset & Deployment"]
)

with overview_tab:
    left_col, right_col = st.columns([1.1, 1.9])

    with left_col:
        st.subheader("Selected Movie")
        st.markdown(f"**Title:** {selected_movie_row['title']}")
        st.markdown(f"**Industry:** {selected_movie_row['industry']}")
        st.markdown(f"**Genre:** {selected_movie_row['genre']}")
        st.markdown(f"**Year:** {selected_movie_row['year']}")
        st.markdown(f"**Director:** {selected_movie_row['director']}")
        st.markdown(f"**Cast:** {selected_movie_row['cast']}")
        st.markdown(f"**IMDb Rating:** {selected_movie_row['rating']}")
        st.info(selected_movie_row["description"] or "No description available.")

    with right_col:
        st.subheader("Top Rated Movies")
        st.dataframe(
            top_rated_movies(movies_df, selected_industry),
            use_container_width=True,
            hide_index=True,
        )

with recommendations_tab:
    st.subheader("Recommended Movies")
    recommendation_df = recommend_movies(
        movies_df,
        similarity_matrix,
        title_to_index,
        selected_movie,
        selected_industry,
        selected_genre,
        selected_year,
        top_n,
    )

    if recommendation_df.empty:
        st.warning("No recommendations match the selected filters. Try broader filters.")
    else:
        st.dataframe(recommendation_df, use_container_width=True, hide_index=True)

with dataset_tab:
    st.subheader("Dataset Details")
    st.write(f"Current source: **{dataset_label}**")
    st.write(
        "The app automatically reads `data/movies.csv`, `data/hollywood_movies.csv`, and "
        "`data/bollywood_movies.csv` when those files are present."
    )
    st.write(
        "You can also upload a CSV at runtime using the sidebar. The required columns are: "
        "`title, industry, genre, year, director, cast, rating, description`."
    )

    st.subheader("How the Model Works")
    st.write(
        "This is a content-based recommender. It combines each movie's industry, genre, director, cast, "
        "and description into one text field, converts that text into TF-IDF vectors, and uses cosine similarity "
        "to find the closest matches."
    )

    with st.expander("Preview Current Dataset"):
        preview_columns = ["title", "industry", "genre", "year", "director", "rating"]
        st.dataframe(movies_df[preview_columns], use_container_width=True, hide_index=True)

    with st.expander("Deployment Notes"):
        st.write("This project is ready for Streamlit Community Cloud and Render.")
        st.code('streamlit run "Recommendation system.py"', language="bash")
