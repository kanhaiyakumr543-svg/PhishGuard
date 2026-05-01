# PhishGuard

PhishGuard is a Flask-based phishing detection project that analyzes both URLs and email content. It includes training scripts for URL and email classifiers, plus a web app for running predictions with the saved model artifacts.

## Features

- URL phishing detection using extracted lexical/domain features and XGBoost.
- Email phishing detection using text cleaning, TF-IDF features, structured metadata, and multiple model options.
- Flask web interface in `app.py`.
- Training scripts for rebuilding email and URL models.

## Project Structure

```text
.
├── app.py
├── templates/
│   └── index.html
├── train_email_model.py
├── train_url_model.py
├── README.md
└── .gitignore
```

Large datasets, trained models, reports, results, and virtual environments are intentionally ignored by Git.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required Python packages:

```powershell
pip install pandas numpy scikit-learn scipy beautifulsoup4 flask xgboost tldextract python-Levenshtein
```

Optional packages for deep learning email models:

```powershell
pip install tensorflow torch transformers
```

## Run the App

```powershell
python app.py
```

Then open the local Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Train Models

Train the email models:

```powershell
python train_email_model.py
```

Train the URL model:

```powershell
python train_url_model.py
```

The training scripts generate model artifacts under `models/` and related output files. These files are not committed because they can be large and can be regenerated.

## Git Notes

The repository ignores:

- `venv/`
- `models/`
- `data/`
- `reports/`
- `results/`
- generated `.csv` datasets
- generated model files such as `.pkl`, `.h5`, `.joblib`, `.onnx`, and `.safetensors`

This keeps `git add .` focused on source code, templates, and documentation.
