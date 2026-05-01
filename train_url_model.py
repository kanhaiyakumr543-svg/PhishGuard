"""
train_url_model.py
Complete training script for URL phishing detection.
"""

import pandas as pd
import numpy as np
import re
import math
import pickle
import warnings
warnings.filterwarnings('ignore')

from urllib.parse import urlparse, parse_qs
import tldextract
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

# ============================================
# 1. FEATURE EXTRACTION
# ============================================

def entropy(text):
    if not text:
        return 0
    text = str(text)
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum([p * math.log(p) / math.log(2.0) for p in prob])

BRANDS = ["google", "paypal", "amazon", "apple", "facebook", 
          "microsoft", "netflix", "instagram", "linkedin"]
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", 
                    "bank", "confirm", "signin", "password", "alert"]
SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".xyz", ".top", ".work", ".date"]

IP_PATTERN = re.compile(r'\d+\.\d+\.\d+\.\d+')
CONSECUTIVE_PATTERN = re.compile(r'(.)\1{2,}')

def extract_url_features(url):
    """Extract 26 features from a single URL"""
    try:
        if pd.isna(url):
            return None
        
        url = str(url).lower().strip()
        url = url.replace("[.]", ".").replace("[d]", ".").replace("(.)", ".")
        
        parse_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(parse_url)
        ext = tldextract.extract(url)
        
        domain = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain
        full_domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        digit_count = sum(c.isdigit() for c in url)
        
        from Levenshtein import distance
        typo_flag = int(any(0 < distance(domain, b) <= 2 for b in BRANDS))
        brand_in_domain = int(any(b in domain for b in BRANDS))
        brand_in_path = int(any(b in path for b in BRANDS))
        
        domain_words = set(re.split(r'[.-]', domain))
        path_words = set(re.split(r'[/.-]', path.strip('/')))
        common_words = domain_words.intersection(path_words)
        domain_path_consistency = len(common_words) / (len(domain_words) + 1)
        
        domain_entropy = entropy(domain)
        path_entropy = entropy(path)
        path_domain_entropy_ratio = path_entropy / domain_entropy if domain_entropy > 0 else 0
        
        features = {
            "url_length": len(url),
            "domain_length": len(full_domain),
            "path_length": len(path),
            "dot_count": url.count("."),
            "hyphen_count": url.count("-"),
            "underscore_count": url.count("_"),
            "slash_count": url.count("/"),
            "digit_count": digit_count,
            "digit_ratio": digit_count / len(url) if len(url) > 0 else 0,
            "special_char_count": sum(not c.isalnum() for c in url),
            "url_entropy": entropy(url),
            "domain_entropy": domain_entropy,
            "subdomain_count": subdomain.count(".") + 1 if subdomain else 0,
            "has_ip": int(bool(IP_PATTERN.search(url))),
            "query_param_count": len(parse_qs(query)),
            "brand_in_domain": brand_in_domain,
            "brand_in_path": brand_in_path,
            "typosquatting": typo_flag,
            "suspicious_word_count": sum(1 for w in SUSPICIOUS_WORDS if w in url),
            "tld_length": len(suffix),
            "suspicious_tld": int(any(suffix == tld.lstrip('.') for tld in SUSPICIOUS_TLDS)),
            "uses_https": int(url.startswith("https")),
            "is_shortened": int(any(s in url for s in ["bit.ly", "tinyurl", "goo.gl", "t.co"])),
            "domain_path_consistency": domain_path_consistency,
            "path_domain_entropy_ratio": path_domain_entropy_ratio,
            "consecutive_chars": int(bool(CONSECUTIVE_PATTERN.search(url)))
        }
        return features
    except Exception as e:
        print(f"Error processing URL: {url} -> {e}")
        return None

# ============================================
# 2. LOAD DATA
# ============================================

print("Loading URL dataset...")
df = pd.read_csv("balanced_urls.csv")
df = df.dropna(subset=["url"])
df = df.reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Extract features
print("Extracting URL features...")
feature_rows = df["url"].apply(extract_url_features)
feature_rows = feature_rows.dropna()
features_df = pd.DataFrame(feature_rows.tolist())
features_df["label"] = df.loc[feature_rows.index, "label"].values

print(f"Features shape: {features_df.shape}")
# ============================================
# SAVE FEATURE DATASET (ADD THIS)
# ============================================

os.makedirs("data", exist_ok=True)

features_df.to_csv("data/url_features_dataset.csv", index=False)

print("✅ URL feature dataset saved to data/url_features_dataset.csv")
# ============================================
# 3. SPLIT AND SCALE
# ============================================

X = features_df.drop("label", axis=1)
y = features_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# FIX BUG #6: Store column order
ALL_FEATURE_NAMES = X.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=ALL_FEATURE_NAMES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=ALL_FEATURE_NAMES)

# ============================================
# 4. FEATURE SELECTION
# ============================================

print("Performing feature selection...")
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train_scaled, y_train)

importances = rf_temp.feature_importances_
importance_df = pd.DataFrame({
    "Feature": ALL_FEATURE_NAMES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

SELECTED_FEATURES = importance_df.head(15)["Feature"].tolist()
print(f"\nSelected {len(SELECTED_FEATURES)} features:")
for i, (feat, imp) in enumerate(zip(SELECTED_FEATURES, importance_df.head(15)["Importance"]), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

X_train_selected = X_train_scaled[SELECTED_FEATURES]
X_test_selected = X_test_scaled[SELECTED_FEATURES]

# ============================================
# 5. TRAIN XGBOOST
# ============================================

print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_selected, y_train)

y_pred = xgb_model.predict(X_test_selected)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n===== OVERALL MODEL PERFORMANCE =====")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ============================================
# 6. SAVE MODELS
# ============================================

os.makedirs('models', exist_ok=True)

with open('models/url_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/url_all_feature_names.pkl', 'wb') as f:
    pickle.dump(ALL_FEATURE_NAMES, f)

with open('models/url_selected_features.pkl', 'wb') as f:
    pickle.dump(SELECTED_FEATURES, f)

xgb_model.save_model('models/url_xgb_model.json')

print("\n URL models saved to 'models/' directory!")