# ReviewLens — Final Project

This package is configured for the exact dataset uploaded for this project:

`DatafinitiElectronicsProductData.csv`

The CSV contains 7,299 rows and the exact review fields used by the app:
- `name`
- `asins`
- `brand`
- `reviews.text`
- `reviews.title`
- `reviews.rating`
- `reviews.doRecommend`
- `reviews.numHelpful`

After excluding rows without usable rating/review text, the app analyzes the remaining reviews.

## Run

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

The dataset is already included in `data/`, so you do not need to download or rename another CSV.

## Project
Product selection → review filtering → text preprocessing → TF-IDF → Logistic Regression → sentiment percentages → rating analysis → common themes → purchase analysis → review explorer.

Sentiment labels:
- 1–2 = Negative
- 3 = Neutral
- 4–5 = Positive

The final purchase analysis is a transparent rule based on aggregate rating and model-predicted sentiment; it is not a separate purchase-intention model.
