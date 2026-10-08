# Cinnamon Quality Classification (Django + Machine Learning)

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py          # trains 5 models, saves ml/artifacts.joblib
python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000/ , register, log in, and predict.

## Use your real dataset
Put a CSV at `data/cinnamon.csv` with columns:
`moisture, ash, volatile_oil, acid_insoluble_ash, chromium, coumarin, quality`
(quality = High / Medium / Low), then run `python train_model.py` again.
Without it, a synthetic demo dataset is used.
