"""
app.py  ─  Phishing Detection Web Application
Supports: URL analysis + Email analysis
Models  : XGBoost (URL), NB/LR/CNN/LSTM/DistilBERT (Email)
"""

import os, re, math, pickle, warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs
import tldextract
from flask import Flask, request, jsonify, render_template_string
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
from bs4 import BeautifulSoup

# ── optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available – CNN/LSTM disabled.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False
    logger.warning("Transformers not available – DistilBERT disabled.")

try:
    from Levenshtein import distance as lev_distance
    LEV_AVAILABLE = True
except Exception:
    LEV_AVAILABLE = False

# =============================================================================
#  CONSTANTS
# =============================================================================

BRANDS = ["google","paypal","amazon","apple","facebook",
          "microsoft","netflix","instagram","linkedin"]

TRUSTED_DOMAINS = [
    "google.com",
    "scholar.google.com",
    "amazon.com",
    "facebook.com",
    "microsoft.com",
    "apple.com"
]
SUSPICIOUS_WORDS = ["login","verify","secure","account","update",
                    "bank","confirm","signin","password","alert"]
SUSPICIOUS_TLDS  = [".tk",".ml",".ga",".cf",".xyz",".top",".work",".date"]
PHISHING_KW_DEFAULT = ["verify","account","login","password","bank",
                        "paypal","confirm","secure","urgent"]

IP_PATTERN          = re.compile(r'\d+\.\d+\.\d+\.\d+')
CONSECUTIVE_PATTERN = re.compile(r'(.)\1{2,}')

MAX_LEN  = 200          # Keras pad length
MAX_WORDS = 10000       # Keras tokenizer vocab

# =============================================================================
#  FEATURE HELPERS
# =============================================================================

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    text = str(text)
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
    return -sum(p * math.log(p) / math.log(2.0) for p in prob)


def extract_url_features(url: str) -> dict | None:
    """26-feature extractor matching training script."""
    try:
        if pd.isna(url):
            return None
        url = str(url).lower().strip()
        url = url.replace("[.]", ".").replace("[d]", ".").replace("(.)", ".")
        parse_url = url if url.startswith(("http://","https://")) else "http://" + url
        parsed    = urlparse(parse_url)
        ext       = tldextract.extract(url)

        domain      = ext.domain
        suffix      = ext.suffix
        subdomain   = ext.subdomain
        full_domain = parsed.netloc
        path        = parsed.path
        query       = parsed.query

        digit_count = sum(c.isdigit() for c in url)

        if LEV_AVAILABLE:
            typo_flag = int(any(0 < lev_distance(domain, b) <= 2 for b in BRANDS))
        else:
            typo_flag = 0

        brand_in_domain = int(any(b in domain for b in BRANDS))
        brand_in_path   = int(any(b in path   for b in BRANDS))

        domain_words = set(re.split(r'[.-]', domain))
        path_words   = set(re.split(r'[/.-]', path.strip('/')))
        common_words = domain_words.intersection(path_words)
        domain_path_consistency = len(common_words) / (len(domain_words) + 1)

        domain_entropy = _entropy(domain)
        path_entropy   = _entropy(path)
        path_domain_entropy_ratio = (path_entropy / domain_entropy
                                     if domain_entropy > 0 else 0)

        return {
            "url_length"               : len(url),
            "domain_length"            : len(full_domain),
            "path_length"              : len(path),
            "dot_count"                : url.count("."),
            "hyphen_count"             : url.count("-"),
            "underscore_count"         : url.count("_"),
            "slash_count"              : url.count("/"),
            "digit_count"              : digit_count,
            "digit_ratio"              : digit_count / len(url) if url else 0,
            "special_char_count"       : sum(not c.isalnum() for c in url),
            "url_entropy"              : _entropy(url),
            "domain_entropy"           : domain_entropy,
            "subdomain_count"          : subdomain.count(".") + 1 if subdomain else 0,
            "has_ip"                   : int(bool(IP_PATTERN.search(url))),
            "query_param_count"        : len(parse_qs(query)),
            "brand_in_domain"          : brand_in_domain,
            "brand_in_path"            : brand_in_path,
            "typosquatting"            : typo_flag,
            "suspicious_word_count"    : sum(1 for w in SUSPICIOUS_WORDS if w in url),
            "tld_length"               : len(suffix),
            "suspicious_tld"           : int(any(suffix == t.lstrip('.') for t in SUSPICIOUS_TLDS)),
            "uses_https"               : int(url.startswith("https")),
            "is_shortened"             : int(any(s in url for s in ["bit.ly","tinyurl","goo.gl","t.co"])),
            "domain_path_consistency"  : domain_path_consistency,
            "path_domain_entropy_ratio": path_domain_entropy_ratio,
            "consecutive_chars"        : int(bool(CONSECUTIVE_PATTERN.search(url))),
        }
    except Exception as e:
        logger.error(f"URL feature error: {e}")
        return None


def clean_email_text(text: str) -> str:
    try:
        if pd.isna(text):
            return ""
        text = str(text)
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            pass
        text = re.sub(r'http\S+', 'URL', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return text.lower()
    except Exception:
        return ""


def extract_email_structured(row: dict, phishing_keywords: list) -> dict:
    """Build the same structured feature vector used at training time."""
    sender   = str(row.get("sender", ""))
    urls_raw = str(row.get("urls", ""))
    subject  = str(row.get("subject", ""))
    raw_text = str(row.get("email_text", ""))
    clean    = str(row.get("clean_text", ""))
    date_str = str(row.get("date", ""))

    sender_domain_length = len(sender.split('@')[-1]) if '@' in sender else 0
    url_count  = len(urls_raw.split())
    url_length = len(urls_raw)
    url_has_ip = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', urls_raw)))
    word_count = len(clean.split())
    char_count = len(clean)
    email_length = len(clean)
    uppercase_ratio = sum(1 for c in raw_text if c.isupper()) / (len(raw_text) + 1)
    exclamation_count = raw_text.count('!')
    keyword_count = sum(1 for kw in phishing_keywords if kw in clean.lower())

    try:
        hour_sent = pd.to_datetime(date_str, errors='coerce', utc=True)
        hour_sent = hour_sent.hour if not pd.isna(hour_sent) else 12
    except Exception:
        hour_sent = 12

    sub_words  = set(subject.lower().split())
    body_words = set(clean.lower().split())
    union = sub_words | body_words
    subject_body_sim = len(sub_words & body_words) / (len(union) + 1)
    url_density = url_count / (word_count + 1)

    return {
        "sender_domain_length": sender_domain_length,
        "url_count"           : url_count,
        "url_length"          : url_length,
        "url_has_ip"          : url_has_ip,
        "word_count"          : word_count,
        "char_count"          : char_count,
        "email_length"        : email_length,
        "uppercase_ratio"     : uppercase_ratio,
        "exclamation_count"   : exclamation_count,
        "keyword_count"       : keyword_count,
        "hour_sent"           : hour_sent,
        "subject_body_sim"    : subject_body_sim,
        "url_density"         : url_density,
    }

# =============================================================================
#  MODEL LOADER
# =============================================================================

class ModelStore:
    """Lazy-loads all saved artefacts once at startup."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._load_url_models()
        self._load_email_models()

    # ── URL ──────────────────────────────────────────────────────────────────
    def _load_url_models(self):
        md = self.model_dir
        try:
            with open(f"{md}/url_scaler.pkl", "rb") as f:
                self.url_scaler = pickle.load(f)
            with open(f"{md}/url_all_feature_names.pkl", "rb") as f:
                self.url_all_features = pickle.load(f)
            with open(f"{md}/url_selected_features.pkl", "rb") as f:
                self.url_selected = pickle.load(f)
            self.url_xgb = XGBClassifier()
            self.url_xgb.load_model(f"{md}/url_xgb_model.json")
            logger.info("URL models loaded ✓")
            self.url_ready = True
        except Exception as e:
            logger.warning(f"URL models not loaded: {e}")
            self.url_ready = False

    # ── Email ─────────────────────────────────────────────────────────────────
    def _load_email_models(self):
        md = self.model_dir
        self.email_nb = self.email_lr = None
        self.email_vectorizer = self.email_structured_cols = None
        self.email_phishing_kw = PHISHING_KW_DEFAULT
        self.email_cnn = self.email_cnn_tok = None
        self.email_lstm = self.email_lstm_tok = None
        self.bert_model = self.bert_tok = None

        try:
            with open(f"{md}/email_vectorizer.pkl", "rb") as f:
                self.email_vectorizer = pickle.load(f)
            with open(f"{md}/email_nb_model.pkl", "rb") as f:
                self.email_nb = pickle.load(f)
            with open(f"{md}/email_lr_model.pkl", "rb") as f:
                self.email_lr = pickle.load(f)
            with open(f"{md}/email_structured_cols.pkl", "rb") as f:
                self.email_structured_cols = pickle.load(f)
            with open(f"{md}/email_phishing_keywords.pkl", "rb") as f:
                self.email_phishing_kw = pickle.load(f)
            logger.info("Email NB/LR models loaded ✓")
        except Exception as e:
            logger.warning(f"Email NB/LR models not loaded: {e}")

        if TF_AVAILABLE:
            try:
                self.email_cnn = load_model(f"{md}/email_cnn_model.h5")
                with open(f"{md}/email_cnn_tokenizer.pkl", "rb") as f:
                    self.email_cnn_tok = pickle.load(f)
                logger.info("CNN model loaded ✓")
            except Exception as e:
                logger.warning(f"CNN model not loaded: {e}")

            try:
                self.email_lstm = load_model(f"{md}/email_lstm_model.h5")
                with open(f"{md}/email_lstm_tokenizer.pkl", "rb") as f:
                    self.email_lstm_tok = pickle.load(f)
                logger.info("LSTM model loaded ✓")
            except Exception as e:
                logger.warning(f"LSTM model not loaded: {e}")

        if BERT_AVAILABLE:
            try:
                self.bert_tok = AutoTokenizer.from_pretrained(f"{md}/email_bert_model")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    f"{md}/email_bert_model"
                )
                self.bert_model.eval()
                logger.info("DistilBERT model loaded ✓")
            except Exception as e:
                logger.warning(f"DistilBERT model not loaded: {e}")

        self.email_ready = self.email_nb is not None

# =============================================================================
#  PREDICTION LOGIC
# =============================================================================

def predict_url(store: ModelStore, url: str) -> dict:
    if not store.url_ready:
        return {"error": "URL models not available"}

    # Normalize URL
    url = url.strip().lower()

    # Extract domain
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    domain = parsed.netloc

    # Whitelist check
    for trusted in TRUSTED_DOMAINS:
        if trusted in domain:
            return {
                "model": "Whitelist",
                "prediction": "Legitimate",
                "confidence": 99.0,
                "label": 0,
                "features": {}
            }

    # Continue ML prediction
    feats = extract_url_features(url)
    if feats is None:
        return {"error": "Feature extraction failed"}

    row = pd.DataFrame([feats])[store.url_all_features]

    scaled = pd.DataFrame(
        store.url_scaler.transform(row),
        columns=store.url_all_features
    )

    X = scaled[store.url_selected]

    prob = float(store.url_xgb.predict_proba(X)[0][1])

    # 🔥 FIXED THRESHOLD
    threshold = 0.8
    label = int(prob >= threshold)

    return {
        "model": "XGBoost",
        "prediction": "Phishing" if label else "Legitimate",
        "confidence": round(prob * 100, 2),
        "label": label,
        "features": {
            k: round(float(v), 4)
            for k, v in feats.items()
            if k in store.url_selected
        }
    }



def predict_email(store: ModelStore, payload: dict) -> dict:
    """
    payload keys: sender, subject, body, urls, date (all optional except body)
    Returns ensemble result plus per-model breakdown.
    """
    if not store.email_ready:
        return {"error": "Email models not available"}

    email_text  = (payload.get("subject","") + " " + payload.get("body","")).strip()
    clean_text  = clean_email_text(email_text)
    row         = {**payload, "email_text": email_text, "clean_text": clean_text}
    struct_dict = extract_email_structured(row, store.email_phishing_kw)

    results  = {}
    probs    = []

    # ── NB + LR ──────────────────────────────────────────────────────────────
    if store.email_vectorizer and store.email_nb and store.email_lr:
        X_text   = store.email_vectorizer.transform([clean_text])
        struct_df = pd.DataFrame([struct_dict])

        # Align columns to training order
        if store.email_structured_cols:
            for col in store.email_structured_cols:
                if col not in struct_df.columns:
                    struct_df[col] = 0
            struct_df = struct_df[store.email_structured_cols]

        X_struct = csr_matrix(struct_df.values.astype(np.float32))
        X_comb   = hstack([X_text, X_struct])

        nb_p  = float(store.email_nb.predict_proba(X_comb)[0][1])
        lr_p  = float(store.email_lr.predict_proba(X_comb)[0][1])
        results["Naive Bayes"]          = round(nb_p * 100, 2)
        results["Logistic Regression"]  = round(lr_p * 100, 2)
        probs += [nb_p, lr_p]

    # ── CNN ───────────────────────────────────────────────────────────────────
    if store.email_cnn and store.email_cnn_tok:
        try:
            seq = store.email_cnn_tok.texts_to_sequences([clean_text])
            pad = pad_sequences(seq, maxlen=MAX_LEN)
            cnn_p = float(store.email_cnn.predict(pad, verbose=0)[0][0])
            results["CNN"]  = round(cnn_p * 100, 2)
            probs.append(cnn_p)
        except Exception as e:
            logger.warning(f"CNN predict error: {e}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if store.email_lstm and store.email_lstm_tok:
        try:
            seq = store.email_lstm_tok.texts_to_sequences([clean_text])
            pad = pad_sequences(seq, maxlen=MAX_LEN)
            lstm_p = float(store.email_lstm.predict(pad, verbose=0)[0][0])
            results["LSTM"] = round(lstm_p * 100, 2)
            probs.append(lstm_p)
        except Exception as e:
            logger.warning(f"LSTM predict error: {e}")

    # ── DistilBERT ────────────────────────────────────────────────────────────
    if store.bert_model and store.bert_tok:
        try:
            enc = store.bert_tok(
                clean_text[:512], truncation=True,
                padding=True, max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                logits = store.bert_model(**enc).logits
            bert_p = float(torch.softmax(logits, dim=1)[0][1])
            results["DistilBERT"] = round(bert_p * 100, 2)
            probs.append(bert_p)
        except Exception as e:
            logger.warning(f"BERT predict error: {e}")

    if not probs:
        return {"error": "No email models produced output"}

    ensemble = float(np.mean(probs))
    label    = int(ensemble >= 0.5)

    return {
        "ensemble_confidence" : round(ensemble * 100, 2),
        "prediction"          : "Phishing" if label else "Legitimate",
        "label"               : label,
        "model_breakdown"     : results,
        "models_used"         : list(results.keys()),
    }

# =============================================================================
#  FLASK APP
# =============================================================================

app   = Flask(__name__)
store = ModelStore(model_dir=os.environ.get("MODEL_DIR", "models"))

# =============================================================================
#  HTML TEMPLATE
# =============================================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhishGuard — Phishing Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #090c10;
    --panel:    #0e1117;
    --border:   #1e2535;
    --text:     #dde4f0;
    --muted:    #5a6480;
    --accent:   #00e5ff;
    --danger:   #ff4b6e;
    --safe:     #00ffb3;
    --warn:     #ffd166;
    --glow-a:   rgba(0,229,255,.18);
    --glow-d:   rgba(255,75,110,.18);
    --radius:   12px;
    --font-h:   'Syne', sans-serif;
    --font-m:   'DM Mono', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-m);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ─── GRID NOISE BG ──────────────────────────────── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(0,229,255,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  /* ─── HEADER ─────────────────────────────────────── */
  header {
    position: relative; z-index: 10;
    display: flex; align-items: center; gap: 16px;
    padding: 28px 40px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #0a0f1a 0%, #0c1220 100%);
  }
  .logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), #0077ff);
    border-radius: 10px;
    display: grid; place-items: center;
    font-size: 22px;
    box-shadow: 0 0 20px var(--glow-a);
    flex-shrink: 0;
  }
  header h1 {
    font-family: var(--font-h);
    font-size: 1.6rem; font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--accent), #7cf0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  header p {
    color: var(--muted); font-size: .75rem; margin-left: auto;
    letter-spacing: .05em; text-transform: uppercase;
  }

  /* ─── TABS ───────────────────────────────────────── */
  .tabs {
    position: relative; z-index: 10;
    display: flex; gap: 0;
    padding: 24px 40px 0;
  }
  .tab {
    padding: 10px 28px;
    border: 1px solid var(--border);
    border-bottom: none;
    background: none; color: var(--muted);
    font-family: var(--font-h); font-size: .85rem; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase;
    cursor: pointer;
    border-radius: var(--radius) var(--radius) 0 0;
    transition: all .2s;
  }
  .tab.active {
    background: var(--panel);
    color: var(--accent);
    border-color: var(--border);
    border-bottom-color: var(--panel);
  }
  .tab:hover:not(.active) { color: var(--text); background: rgba(255,255,255,.03); }

  /* ─── MAIN ───────────────────────────────────────── */
  main {
    position: relative; z-index: 10;
    max-width: 900px; margin: 0 auto;
    padding: 0 40px 60px;
  }

  .pane {
    display: none;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 0 var(--radius) var(--radius) var(--radius);
    padding: 32px 36px;
    animation: fadeIn .3s ease;
  }
  .pane.active { display: block; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(6px); } to { opacity:1; } }

  /* ─── FORM ELEMENTS ──────────────────────────────── */
  label {
    display: block;
    font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 8px; margin-top: 20px;
  }
  label:first-child { margin-top: 0; }

  input[type=text], textarea {
    width: 100%;
    background: #080b10;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font-m); font-size: .9rem;
    padding: 12px 16px;
    outline: none;
    transition: border-color .2s, box-shadow .2s;
    resize: vertical;
  }
  input[type=text]:focus, textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--glow-a);
  }

  .btn {
    margin-top: 24px;
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #00c6e8, #0057c8);
    border: none; border-radius: 8px;
    color: #fff; font-family: var(--font-h);
    font-size: .95rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    cursor: pointer;
    transition: opacity .2s, transform .15s, box-shadow .2s;
    box-shadow: 0 4px 20px rgba(0,180,230,.25);
  }
  .btn:hover { opacity:.9; transform: translateY(-1px); box-shadow: 0 6px 28px rgba(0,180,230,.35); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity:.45; cursor:not-allowed; transform:none; }

  /* ─── RESULT CARD ────────────────────────────────── */
  #url-result, #email-result {
    margin-top: 28px;
    display: none;
  }

  .result-card {
    border-radius: var(--radius);
    border: 1px solid;
    padding: 24px 28px;
    animation: fadeIn .35s ease;
  }
  .result-card.phishing {
    border-color: var(--danger);
    background: rgba(255,75,110,.07);
    box-shadow: 0 0 30px var(--glow-d);
  }
  .result-card.safe {
    border-color: var(--safe);
    background: rgba(0,255,179,.06);
    box-shadow: 0 0 30px rgba(0,255,179,.12);
  }

  .verdict {
    display: flex; align-items: center; gap: 14px; margin-bottom: 16px;
  }
  .verdict-icon { font-size: 2.2rem; }
  .verdict-text { font-family: var(--font-h); font-size: 1.6rem; font-weight: 800; }
  .verdict-text.phishing { color: var(--danger); }
  .verdict-text.safe     { color: var(--safe);   }

  .conf-row {
    display: flex; align-items: center; gap: 16px; margin-bottom: 20px;
  }
  .conf-bar-wrap {
    flex: 1; height: 8px; background: var(--border); border-radius: 999px; overflow: hidden;
  }
  .conf-bar {
    height: 100%; border-radius: 999px;
    transition: width .6s cubic-bezier(.22,1,.36,1);
  }
  .conf-bar.phishing { background: linear-gradient(90deg, var(--warn), var(--danger)); }
  .conf-bar.safe     { background: linear-gradient(90deg, var(--accent), var(--safe)); }
  .conf-label { font-size: .8rem; color: var(--muted); white-space: nowrap; }

  /* ─── MODEL BREAKDOWN ────────────────────────────── */
  .breakdown-title {
    font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 12px; margin-top: 8px;
  }
  .model-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
  }
  .model-name {
    width: 140px; flex-shrink: 0;
    font-size: .78rem; color: var(--text);
  }
  .model-bar-wrap {
    flex: 1; height: 6px; background: var(--border); border-radius: 999px; overflow: hidden;
  }
  .model-bar {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #0057c8, var(--accent));
    transition: width .5s ease;
  }
  .model-pct {
    width: 48px; text-align: right;
    font-size: .78rem; color: var(--muted);
  }

  /* ─── FEATURE TABLE ──────────────────────────────── */
  .feats-toggle {
    background: none; border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted);
    font-family: var(--font-m); font-size: .78rem;
    padding: 6px 14px; cursor: pointer;
    margin-top: 16px;
    transition: color .2s, border-color .2s;
  }
  .feats-toggle:hover { color: var(--accent); border-color: var(--accent); }

  .feats-grid {
    display: none;
    margin-top: 14px;
    display: none;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }
  .feats-grid.open { display: grid; }
  .feat-chip {
    background: #0b0f18; border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px;
    font-size: .74rem;
  }
  .feat-chip span { color: var(--accent); float: right; }

  /* ─── SPINNER ────────────────────────────────────── */
  .spinner {
    display: none; align-items: center; justify-content: center; gap: 12px;
    padding: 20px 0; color: var(--muted); font-size: .84rem;
  }
  .spinner.show { display: flex; }
  .spin-ring {
    width: 22px; height: 22px; border-radius: 50%;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ─── STATUS BADGES ──────────────────────────────── */
  .model-status {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; margin-bottom: 2px;
  }
  .badge {
    font-size: .68rem; letter-spacing: .06em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 999px;
    border: 1px solid;
  }
  .badge.ok    { border-color: var(--safe);   color: var(--safe);   background: rgba(0,255,179,.08); }
  .badge.off   { border-color: var(--muted);  color: var(--muted);  background: transparent; }

  /* ─── RESPONSIVE ─────────────────────────────────── */
  @media (max-width: 600px) {
    header, main, .tabs { padding-left: 16px; padding-right: 16px; }
    .pane { padding: 20px 18px; }
    .model-name { width: 100px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-icon">🛡</div>
  <div>
    <h1>PhishGuard</h1>
  </div>
  <p>AI Phishing Detection</p>
</header>

<div class="tabs" style="max-width:900px;margin:0 auto;padding-left:40px;padding-right:40px;">
  <button class="tab active" onclick="switchTab('url',this)">🔗 URL Analysis</button>
  <button class="tab"        onclick="switchTab('email',this)">✉️ Email Analysis</button>
</div>

<main>

<!-- ═══ URL PANE ═══════════════════════════════════════════════════ -->
<div id="pane-url" class="pane active">
  <label>URL to Analyse</label>
  <input type="text" id="url-input" placeholder="https://example.com/login?verify=account"
         onkeydown="if(event.key==='Enter') analyseURL()">
  <button class="btn" id="url-btn" onclick="analyseURL()">Scan URL</button>

  <div class="spinner" id="url-spin"><div class="spin-ring"></div>Extracting features &amp; scoring…</div>

  <div id="url-result"></div>
</div>

<!-- ═══ EMAIL PANE ══════════════════════════════════════════════════ -->
<div id="pane-email" class="pane">
  <label>Sender (optional)</label>
  <input type="text" id="e-sender" placeholder="noreply@support-paypal.xyz">

  <label>Subject</label>
  <input type="text" id="e-subject" placeholder="Urgent: Verify your account immediately!">

  <label>Body</label>
  <textarea id="e-body" rows="6" placeholder="Dear customer, your account has been suspended…"></textarea>

  <label>URLs in email (space-separated, optional)</label>
  <input type="text" id="e-urls" placeholder="http://paypa1-verify.tk/login">

  <label>Date sent (optional)</label>
  <input type="text" id="e-date" placeholder="2024-03-15T03:41:00Z">

  <button class="btn" id="email-btn" onclick="analyseEmail()">Analyse Email</button>
  <div class="spinner" id="email-spin"><div class="spin-ring"></div>Running ensemble models…</div>

  <div id="email-result"></div>
</div>

</main>

<script>
/* ── tab switch ──────────────────────────────────────── */
function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('pane-' + name).classList.add('active');
}

/* ── render result card ──────────────────────────────── */
function renderVerdict(container, isPhishing, conf, extra) {
  const cls   = isPhishing ? 'phishing' : 'safe';
  const icon  = isPhishing ? '⚠️' : '✅';
  const label = isPhishing ? 'Phishing Detected' : 'Looks Legitimate';

  let breakdownHTML = '';
  if (extra.model_breakdown) {
    const entries = Object.entries(extra.model_breakdown);
    breakdownHTML = `<p class="breakdown-title">Model Breakdown</p>`;
    entries.forEach(([name, pct]) => {
      breakdownHTML += `
        <div class="model-row">
          <div class="model-name">${name}</div>
          <div class="model-bar-wrap"><div class="model-bar" style="width:${pct}%"></div></div>
          <div class="model-pct">${pct}%</div>
        </div>`;
    });
  }

  let featsHTML = '';
  if (extra.features) {
    const chips = Object.entries(extra.features)
      .map(([k,v]) => `<div class="feat-chip">${k}<span>${v}</span></div>`)
      .join('');
    featsHTML = `
      <button class="feats-toggle" onclick="toggleFeats(this)">▶ Show feature values</button>
      <div class="feats-grid">${chips}</div>`;
  }

  container.style.display = 'block';
  container.innerHTML = `
    <div class="result-card ${cls}">
      <div class="verdict">
        <div class="verdict-icon">${icon}</div>
        <div class="verdict-text ${cls}">${label}</div>
      </div>
      <div class="conf-row">
        <div class="conf-bar-wrap">
          <div class="conf-bar ${cls}" style="width:${conf}%"></div>
        </div>
        <div class="conf-label">Confidence: <b>${conf}%</b></div>
      </div>
      ${extra.model ? `<p style="font-size:.78rem;color:var(--muted);margin-bottom:12px">Model: ${extra.model}</p>` : ''}
      ${breakdownHTML}
      ${extra.models_used ? `<p style="font-size:.76rem;color:var(--muted);margin-top:10px">Ensemble of: ${extra.models_used.join(' · ')}</p>` : ''}
      ${featsHTML}
    </div>`;
}

function toggleFeats(btn) {
  const grid = btn.nextElementSibling;
  const open = grid.classList.toggle('open');
  btn.textContent = (open ? '▼ Hide' : '▶ Show') + ' feature values';
}

/* ── URL analysis ────────────────────────────────────── */
async function analyseURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { alert('Please enter a URL.'); return; }

  const btn  = document.getElementById('url-btn');
  const spin = document.getElementById('url-spin');
  const res  = document.getElementById('url-result');

  btn.disabled = true;
  spin.classList.add('show');
  res.style.display = 'none';

  try {
    const r    = await fetch('/api/predict/url', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ url })
    });
    const data = await r.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderVerdict(res, data.label === 1, data.confidence, data);
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    spin.classList.remove('show');
  }
}

/* ── Email analysis ──────────────────────────────────── */
async function analyseEmail() {
  const body = document.getElementById('e-body').value.trim();
  if (!body) { alert('Please enter the email body.'); return; }

  const btn  = document.getElementById('email-btn');
  const spin = document.getElementById('email-spin');
  const res  = document.getElementById('email-result');

  btn.disabled = true;
  spin.classList.add('show');
  res.style.display = 'none';

  const payload = {
    sender : document.getElementById('e-sender').value,
    subject: document.getElementById('e-subject').value,
    body,
    urls   : document.getElementById('e-urls').value,
    date   : document.getElementById('e-date').value,
  };

  try {
    const r    = await fetch('/api/predict/email', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderVerdict(res, data.label === 1, data.ensemble_confidence, data);
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    spin.classList.remove('show');
  }
}
</script>
</body>
</html>
"""

# =============================================================================
#  ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict/url", methods=["POST"])
def api_url():
    data = request.get_json(force=True)
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    return jsonify(predict_url(store, url))


@app.route("/api/predict/email", methods=["POST"])
def api_email():
    payload = request.get_json(force=True)
    if not payload.get("body","").strip():
        return jsonify({"error": "Email body is required"}), 400
    return jsonify(predict_email(store, payload))


@app.route("/api/status")
def api_status():
    return jsonify({
        "url_models_ready"   : store.url_ready,
        "email_models_ready" : store.email_ready,
        "cnn_ready"          : store.email_cnn  is not None,
        "lstm_ready"         : store.email_lstm is not None,
        "bert_ready"         : store.bert_model is not None,
    })


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info(f"Starting PhishGuard on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)