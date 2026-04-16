# AgentMod

AgentMod is an agentic AI moderation system that combines a from-scratch neural network and from-scratch K-means clustering to decide whether comments are allowed, flagged, or auto-removed.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download Jigsaw data from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
3. Place `train.csv` in `data/`.

## Run Notebook

```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```

## Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

## Important Note

The notebook must be run to completion first so trained artifacts are saved in `models/`. The Streamlit app depends on these saved model files and benchmark outputs.
