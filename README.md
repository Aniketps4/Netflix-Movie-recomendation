# Netflix-Movie-recomendation
# Netflix Movie Recommendation System

## Overview

This is a **Netflix Movie Recommendation System** built using **Streamlit** and **Machine Learning**. It recommends similar TV shows based on user selection, utilizing **Principal Component Analysis (PCA)** for dimensionality reduction and **Nearest Neighbors (NN)** for finding similar items.

## Features

- Load Netflix TV Shows dataset
- Preprocess data (Label Encoding, Scaling, PCA)
- Find similar shows based on cosine similarity
- Streamlit-based interactive UI for recommendations

## Installation

### Prerequisites

Ensure you have **Python 3.x** installed.

### Install Dependencies

Run the following command to install required libraries:

```bash
pip install streamlit pandas numpy scikit-learn
```

## How to Run

1. Clone the repository or copy the project files.
2. Ensure the dataset **"TV Shows - Netflix.csv"** is present.
3. Run the Streamlit app:
   ```bash
   streamlit run netflix_recommender.py
   ```
4. Select a TV Show from the dropdown and get recommendations.

## File Structure

```
Netflix-Recommender/
│── TV Shows - Netflix.csv  # Dataset
│── netflix_recommender.py  # Streamlit App Code
│── requirements.txt        # Dependencies
│── README.md               # Project Documentation
```

## Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`

## Future Improvements

- Expand recommendations to include **movies**.
- Enhance similarity calculations with **deep learning models**.
- Improve UI with **better visuals** and filters.

## Author

Developed by 

Aniket Pasi

