# Movie Recommendation System

A deployable Netflix-style movie recommendation web app built with Python, Streamlit, pandas, and scikit-learn.

## Features

- Content-based recommendation using TF-IDF
- Cosine similarity to find similar movies
- Netflix-style dark streaming interface
- Hollywood and Bollywood support
- Industry, genre, and release-year filters
- Runtime CSV upload and downloadable template
- Featured hero banner and movie spotlight
- Movie poster display for every title
- Personalized recommendation rows like "Top Picks For You"
- Trending, new release, and top-rated browsing sections
- Dataset preview and detailed recommendation table
- Includes a larger imported Bollywood movie dataset
- Ready for local run and cloud deployment

## Project File

- `Recommendation system.py`: main Streamlit application
- `data/movies.csv`: combined movie catalog used by the recommender
- `data/hollywood_movies.csv`: optional Hollywood-only bulk dataset
- `data/bollywood_movies.csv`: optional Bollywood-only bulk dataset
- `data/hollywood_movies_template.csv`: starter template for Hollywood data
- `data/bollywood_movies_template.csv`: starter template for Bollywood data
- `Procfile`: deployment entry for platforms that support Procfile
- `render.yaml`: Render deployment configuration
- `.streamlit/config.toml`: Streamlit configuration and theme
- `.gitignore`: keeps local and deployment-only files out of Git

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run "Recommendation system.py"
```

3. Optional:

- upload a CSV from the sidebar while the app is running
- or expand the files inside `data/`

## Dataset Format

Use this column order in your CSV files. The last column is optional:

```text
title,industry,genre,year,director,cast,rating,description,poster_url
```

Example row:

```text
Inception,Hollywood,Sci-Fi Action Thriller,2010,Christopher Nolan,"Leonardo DiCaprio, Joseph Gordon-Levitt",8.8,A thief enters dreams to steal secrets and plant ideas.,https://example.com/inception.jpg
```

## Deployment Options

### Streamlit Community Cloud

1. Upload this project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your repository.
4. Set the main file path to `Recommendation system.py`.
5. Deploy.

### Render

This project already includes `render.yaml`. If you want to configure it manually, use:

```bash
streamlit run "Recommendation system.py" --server.port $PORT --server.address 0.0.0.0
```

### Other Platforms

If the platform supports a `Procfile`, this repository already includes one.

## Project Structure

```text
.
|-- Recommendation system.py
|-- requirements.txt
|-- Procfile
|-- render.yaml
|-- .streamlit/
|   `-- config.toml
`-- data/
    |-- movies.csv
    |-- hollywood_movies_template.csv
    `-- bollywood_movies_template.csv
```

## ML Approach

The system is content-based:

- movie metadata is merged into one text field
- TF-IDF converts text into feature vectors
- cosine similarity compares movies
- the highest similarity scores are returned as recommendations

## User Experience

The interface is designed to feel closer to a streaming platform than a plain analytics dashboard. It includes:

- a cinematic featured section for the selected movie
- poster-style movie artwork in each row
- personalized recommendation shelves
- searchable movie browsing
- quick filters for industry, genre, and year
- a detailed table view for comparing recommended titles

## Expand the Dataset

The app now reads from any of these files if they exist:

- `data/movies.csv`
- `data/hollywood_movies.csv`
- `data/bollywood_movies.csv`

This project now includes a generated `data/bollywood_movies.csv` created from the provided Bollywood movie archive.

You can keep adding more Hollywood and Bollywood movies by appending rows with these columns:

```text
title,industry,genre,year,director,cast,rating,description,poster_url
```

Template files are included:

- `data/hollywood_movies_template.csv`
- `data/bollywood_movies_template.csv`

You can also upload a dataset directly in the sidebar without changing project files.
If `poster_url` is left empty, the app generates a poster-style placeholder automatically.

## Notes

- This recommender is content-based, not collaborative filtering.
- Recommendation quality depends on the quality and completeness of your dataset.
- Some imported datasets may use a normalized movie `score` rather than a true IMDb rating.
- To scale to a very large catalog, keep expanding the CSV files in `data/`.
