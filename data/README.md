# Dataset Guide

This folder stores the CSV files used by the movie recommendation app.

Supported files:

- `movies.csv`
- `hollywood_movies.csv`
- `bollywood_movies.csv`

Required columns:

```text
title,industry,genre,year,director,cast,rating,description
```

Optional column:

```text
poster_url
```

Notes:

- `industry` should usually be `Hollywood` or `Bollywood`
- `rating` should be a number between `0` and `10`
- `poster_url` can contain a direct image URL for the movie poster
- if your source dataset does not contain ratings, you can use a normalized score
- duplicate movie titles are automatically removed and the first row is kept
- if no dataset files are present, the app falls back to a small built-in sample
