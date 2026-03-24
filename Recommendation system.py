from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_TITLE = "Movie Recommendation System"
APP_SUBTITLE = "A Netflix-style movie discovery experience powered by content-based machine learning."
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
OPTIONAL_COLUMNS = ["poster_url"]

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
        "poster_url": "",
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
        "poster_url": "",
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
        "poster_url": "",
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
        "poster_url": "",
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
        "poster_url": "",
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
        "poster_url": "",
    },
]


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(181, 24, 24, 0.28), transparent 34%),
                linear-gradient(180deg, #060606 0%, #0f0f10 34%, #151515 100%);
            color: #f5f5f1;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        .hero-card {
            padding: 2rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(0, 0, 0, 0.72), rgba(26, 26, 26, 0.82)),
                linear-gradient(120deg, rgba(229, 9, 20, 0.25), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
        }
        .hero-kicker {
            font-size: 0.78rem;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: #ffb3b3;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.85rem;
            color: #ffffff;
        }
        .hero-description {
            font-size: 1rem;
            line-height: 1.75;
            color: #e6e6e6;
            max-width: 760px;
        }
        .pill-row {
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }
        .pill {
            display: inline-block;
            margin-right: 0.55rem;
            margin-bottom: 0.55rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #f4f4f5;
            font-size: 0.88rem;
        }
        .section-heading {
            font-size: 1.2rem;
            font-weight: 700;
            margin: 1.2rem 0 0.75rem 0;
            color: #ffffff;
        }
        .movie-card {
            min-height: 420px;
            border-radius: 18px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(32, 32, 35, 0.96), rgba(14, 14, 16, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.07);
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
        }
        .movie-card img {
            width: 100%;
            height: 250px;
            object-fit: cover;
            display: block;
            background: #0d0d0d;
        }
        .movie-card-body {
            padding: 1rem;
        }
        .movie-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.35rem;
        }
        .movie-card-meta {
            color: #bfbfbf;
            font-size: 0.88rem;
            margin-bottom: 0.7rem;
        }
        .movie-card-description {
            color: #e7e7e7;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .hero-poster img {
            width: 100%;
            max-width: 320px;
            border-radius: 22px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101113 0%, #17181c 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_industry(value):
    clean_value = str(value).strip().lower()
    mapping = {"hollywood": "Hollywood", "bollywood": "Bollywood"}
    return mapping.get(clean_value, str(value).strip().title())


def build_poster_data_uri(movie):
    title = str(movie["title"]).upper()
    year = str(movie["year"])
    industry = str(movie["industry"])
    rating = str(movie["rating"])
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='600' height='900' viewBox='0 0 600 900'>
        <defs>
            <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
                <stop offset='0%' stop-color='#7f1d1d'/>
                <stop offset='50%' stop-color='#111827'/>
                <stop offset='100%' stop-color='#020617'/>
            </linearGradient>
        </defs>
        <rect width='600' height='900' fill='url(#bg)'/>
        <rect x='30' y='30' width='540' height='840' rx='28' fill='none' stroke='rgba(255,255,255,0.18)' stroke-width='2'/>
        <text x='60' y='110' fill='#fca5a5' font-size='24' font-family='Arial, sans-serif' letter-spacing='5'>FEATURE FILM</text>
        <text x='60' y='180' fill='white' font-size='54' font-weight='700' font-family='Arial, sans-serif'>{title[:22]}</text>
        <text x='60' y='245' fill='#e5e7eb' font-size='30' font-family='Arial, sans-serif'>{industry}</text>
        <text x='60' y='295' fill='#e5e7eb' font-size='30' font-family='Arial, sans-serif'>{year}</text>
        <text x='60' y='345' fill='#fef3c7' font-size='30' font-family='Arial, sans-serif'>Score {rating}</text>
        <circle cx='470' cy='190' r='90' fill='rgba(255,255,255,0.08)'/>
        <circle cx='470' cy='190' r='54' fill='rgba(255,255,255,0.12)'/>
        <path d='M448 156 L510 190 L448 224 Z' fill='white'/>
        <text x='60' y='760' fill='white' font-size='26' font-family='Arial, sans-serif'>Movie Recommendation System</text>
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def get_poster_source(movie):
    poster_url = str(movie.get("poster_url", "")).strip()
    return poster_url if poster_url else build_poster_data_uri(movie)


def sanitize_movie_dataframe(movie_df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in movie_df.columns]
    if missing_columns:
        raise ValueError("Missing required columns in dataset: " + ", ".join(missing_columns))

    movie_df = movie_df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in movie_df.columns:
            movie_df[column] = ""

    selected_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    movie_df = movie_df[selected_columns]

    for column in ["title", "industry", "genre", "director", "cast", "description", "poster_url"]:
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
                "title": movie["title"],
                "industry": movie["industry"],
                "genre": movie["genre"],
                "year": int(movie["year"]),
                "director": movie["director"],
                "rating": float(movie["rating"]),
                "description": movie["description"],
                "poster_url": movie.get("poster_url", ""),
                "score": round(float(score) * 100, 2),
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
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS).to_csv(index=False).encode("utf-8")


def top_rated_movies(movie_df, industry_filter, limit=8):
    filtered_df = movie_df if industry_filter == "All" else movie_df[movie_df["industry"] == industry_filter]
    return filtered_df.sort_values(by=["rating", "year"], ascending=[False, False]).head(limit)


def newest_movies(movie_df, industry_filter, limit=8):
    filtered_df = movie_df if industry_filter == "All" else movie_df[movie_df["industry"] == industry_filter]
    return filtered_df.sort_values(by=["year", "rating"], ascending=[False, False]).head(limit)


def trending_movies(movie_df, industry_filter, limit=8):
    filtered_df = movie_df if industry_filter == "All" else movie_df[movie_df["industry"] == industry_filter]
    trending_df = filtered_df.assign(
        trend_score=(filtered_df["rating"] * 0.7) + ((filtered_df["year"] - filtered_df["year"].min()) * 0.03)
    )
    return trending_df.sort_values(by=["trend_score", "rating"], ascending=[False, False]).head(limit)


def render_hero(movie):
    genres = [genre for genre in str(movie["genre"]).split()[:3] if genre]
    pills = "".join(
        [f"<span class='pill'>{movie['industry']}</span>", f"<span class='pill'>{movie['year']}</span>", f"<span class='pill'>Score {movie['rating']}</span>"]
        + [f"<span class='pill'>{genre}</span>" for genre in genres]
    )
    left_col, right_col = st.columns([1.8, 0.9])
    with left_col:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-kicker">Featured Tonight</div>
                <div class="hero-title">{movie['title']}</div>
                <div class="pill-row">{pills}</div>
                <div class="hero-description">{movie['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div class="hero-poster">
                <img src="{get_poster_source(movie)}" alt="{movie['title']} poster"/>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_movie_row(title, movie_df, show_score=False):
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)
    row_items = movie_df.head(4).to_dict("records")
    columns = st.columns(4)
    for column, movie in zip(columns, row_items):
        score_html = ""
        if show_score and "score" in movie:
            score_html = f"<div class='movie-card-meta'>Match Score: {movie['score']}%</div>"
        with column:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{get_poster_source(movie)}" alt="{movie['title']} poster"/>
                    <div class="movie-card-body">
                        <div class="movie-card-title">{movie['title']}</div>
                        <div class="movie-card-meta">{movie['industry']} | {movie['year']} | Score {movie['rating']}</div>
                        {score_html}
                        <div class="movie-card-description">{movie['description'][:140]}...</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.set_page_config(page_title=APP_TITLE, page_icon=":movie_camera:", layout="wide")
inject_styles()

try:
    base_movies_df, available_files = load_movies()
except Exception as error:
    st.error(f"Dataset error: {error}")
    st.stop()

with st.sidebar:
    st.header("Library")
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
        dataset_label = f"Uploaded: {uploaded_file.name}"
    except Exception as error:
        st.error(f"Uploaded dataset error: {error}")
        st.stop()
else:
    movies_df = base_movies_df
    dataset_label = ", ".join(file_path.name for file_path in available_files) if available_files else "Built-in sample"

similarity_matrix, title_to_index = train_recommender(movies_df)
industry_choices = ["All"] + sorted(movies_df["industry"].dropna().unique().tolist())
genre_choices = ["All"] + sorted({genre for value in movies_df["genre"] for genre in value.split() if genre})

with st.sidebar:
    st.header("Personalize")
    selected_industry = st.selectbox("Industry", industry_choices)
    search_term = st.text_input("Search title", placeholder="Search movies")
    movie_options_df = filter_movie_options(movies_df, selected_industry, search_term)
    if movie_options_df.empty:
        st.warning("No movies match the current search or industry filter.")
        st.stop()
    selected_movie = st.selectbox("Because you watched", sorted(movie_options_df["title"].tolist()))
    selected_genre = st.selectbox("Genre focus", genre_choices)
    selected_year = st.slider(
        "Minimum year",
        min_value=int(movies_df["year"].min()),
        max_value=int(movies_df["year"].max()),
        value=int(movies_df["year"].min()),
    )
    top_n = st.slider("Recommendation count", min_value=4, max_value=16, value=8)
    st.caption(f"Source: {dataset_label}")

selected_movie_row = movies_df.loc[movies_df["title"] == selected_movie].iloc[0]
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

st.markdown(f"# {APP_TITLE}")
st.caption(APP_SUBTITLE)
render_hero(selected_movie_row)

metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Movies Loaded", len(movies_df))
metric_two.metric("Industries", movies_df["industry"].nunique())
metric_three.metric("Genres", len(genre_choices) - 1)
metric_four.metric("For You", len(recommendation_df))

render_movie_row("Because You Watched This", pd.DataFrame([selected_movie_row]))

if recommendation_df.empty:
    st.warning("No recommendations match the current filters. Try broader settings from the sidebar.")
else:
    render_movie_row("Top Picks For You", recommendation_df, show_score=True)

render_movie_row("Trending Now", trending_movies(movies_df, selected_industry))
render_movie_row("New Releases", newest_movies(movies_df, selected_industry))
render_movie_row("Top Rated On Movie Recommendation System", top_rated_movies(movies_df, selected_industry))

details_tab, recommendations_tab, library_tab = st.tabs(
    ["Selected Movie", "Recommendation Table", "Library & Model"]
)

with details_tab:
    left_col, right_col = st.columns([0.8, 1.4])
    with left_col:
        st.image(get_poster_source(selected_movie_row), use_container_width=True)
    with right_col:
        st.subheader(selected_movie_row["title"])
        st.write(f"**Industry:** {selected_movie_row['industry']}")
        st.write(f"**Genre:** {selected_movie_row['genre']}")
        st.write(f"**Year:** {selected_movie_row['year']}")
        st.write(f"**Director:** {selected_movie_row['director']}")
        st.write(f"**Cast:** {selected_movie_row['cast']}")
        st.write(f"**Score:** {selected_movie_row['rating']}")
        st.subheader("Synopsis")
        st.write(selected_movie_row["description"] or "No description available.")

with recommendations_tab:
    st.subheader("Detailed Recommendations")
    if recommendation_df.empty:
        st.info("No recommendation rows are available for the current filter set.")
    else:
        table_df = recommendation_df.rename(
            columns={
                "title": "Title",
                "industry": "Industry",
                "genre": "Genre",
                "year": "Year",
                "director": "Director",
                "rating": "Score",
                "score": "Match %",
                "description": "Description",
                "poster_url": "Poster URL",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

with library_tab:
    st.subheader("Library Details")
    st.write(
        "The app reads from `data/movies.csv`, `data/hollywood_movies.csv`, and "
        "`data/bollywood_movies.csv` when available, or falls back to a built-in starter sample."
    )
    st.write(
        "If your CSV includes a `poster_url` column, the app will show real movie images. "
        "If not, it generates poster-style placeholder images automatically."
    )
    st.write(
        "The recommendation engine combines industry, genre, director, cast, and description into one text field, "
        "then uses TF-IDF and cosine similarity to surface similar titles."
    )
    with st.expander("Preview Library"):
        preview_columns = ["title", "industry", "genre", "year", "director", "rating", "poster_url"]
        st.dataframe(movies_df[preview_columns], use_container_width=True, hide_index=True)
    with st.expander("Deployment Command"):
        st.code('streamlit run "Recommendation system.py"', language="bash")
