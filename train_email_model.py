import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import os

# ============================================
# 1. CLEANING FUNCTIONS
# ============================================

def clean_email_text(text):
    # Safe cleaning (handles broken HTML)
    try:
        if pd.isna(text):
            return ""
        text = str(text)

        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except:
            pass  # skip broken HTML

        text = re.sub(r'http\S+', 'URL', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return text.lower()

    except:
        return ""


def extract_structured_features(df):
    """Extract structured features from email data"""
    
    features = pd.DataFrame(index=df.index)

    # Sender domain length
    features['sender_domain_length'] = df['sender'].apply(
        lambda x: len(str(x).split('@')[-1]) if '@' in str(x) else 0
    )

    # URL features
    features['url_count'] = df['urls'].apply(lambda x: len(str(x).split()))
    features['url_length'] = df['urls'].apply(lambda x: len(str(x)))
    features['url_has_ip'] = df['urls'].apply(
        lambda x: int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', str(x))))
    )

    # Text features
    clean_text = df['clean_text'].fillna('')
    features['word_count'] = clean_text.apply(lambda x: len(str(x).split()))
    features['char_count'] = clean_text.apply(lambda x: len(str(x)))
    features['email_length'] = clean_text.apply(len)

    # Uppercase ratio
    email_text = df['email_text'].fillna('')
    features['uppercase_ratio'] = email_text.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )

    # Exclamation marks
    features['exclamation_count'] = email_text.apply(lambda x: str(x).count('!'))

    # Phishing keywords
    phishing_keywords = ['verify', 'account', 'login', 'password', 'bank', 
                         'paypal', 'confirm', 'secure', 'urgent']

    features['keyword_count'] = clean_text.apply(
        lambda x: sum(1 for kw in phishing_keywords if kw in str(x).lower())
    )

    # Hour sent
    features['hour_sent'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.hour.fillna(12)

    # Subject-body similarity
    features['subject_body_sim'] = df.apply(
        lambda row: len(set(str(row['subject']).lower().split()) & 
                        set(str(row['clean_text']).lower().split())) /
                    (len(set(str(row['subject']).lower().split()) | 
                         set(str(row['clean_text']).lower().split())) + 1),
        axis=1
    )

    # URL density
    features['url_density'] = features['url_count'] / (features['word_count'] + 1)

    return features, phishing_keywords


# ============================================
# 2. LOAD DATA
# ============================================

print("Loading data...")

ceas_df = pd.read_csv("CEAS_08.csv", on_bad_lines='skip', engine='python')
spam_df = pd.read_csv("SpamAssasin.csv")

# Keep needed columns
ceas_df = ceas_df[['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']]
spam_df = spam_df[['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']]

# Combine
data = pd.concat([spam_df, ceas_df], ignore_index=True)
data = data.astype(str)
data = data.dropna()
data = data.drop_duplicates()

# Normalize labels
if data['label'].dtype == 'object':
    data['label'] = data['label'].map(
        lambda x: 1 if str(x).lower() in ['spam', 'phishing', '1'] else 0
    )
else:
    data['label'] = data['label'].astype(int)

print("Dataset loaded:", data.shape)

# Combine subject + body
data['email_text'] = data['subject'].fillna('') + " " + data['body'].fillna('')
data = data[data['email_text'].str.len() < 10000]
# Clean text
data['clean_text'] = data['email_text'].apply(clean_email_text)

# Structured features
structured_features, PHISHING_KEYWORDS = extract_structured_features(data)

print("Structured features:", structured_features.shape)

# ================================
# SAVE DATASETS (ADD HERE)
# ================================

os.makedirs("data", exist_ok=True)

structured_features.to_csv("data/email_structured_features.csv", index=False)
print("Saved structured features dataset")

data.to_csv("data/email_combined_dataset.csv", index=False)
print("Saved combined dataset")

print("Structured features:", structured_features.shape)

# ============================================
# 3. TF-IDF FEATURES
# ============================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3
)

X_text = vectorizer.fit_transform(data['clean_text'])

# ============================================
# 4. COMBINE FEATURES
# ============================================

X_struct = csr_matrix(structured_features.values.astype(np.float32))

X = hstack([X_text, X_struct])
y = data['label'].values

print("Final feature shape:", X.shape)

# ============================================
# 5. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 6. TRAIN MODELS
# ============================================

print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
print("NB Accuracy:", accuracy_score(y_test, nb_pred))

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
print("LR Accuracy:", accuracy_score(y_test, lr_pred))

# ============================================
# PREPARE TEXT DATA FOR DL MODELS
# ============================================

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    data['clean_text'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

# ---------------- DISTILBERT ----------------
print("\nTraining DistilBERT (light version)...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# ⚠️ Use small sample (VERY IMPORTANT)
sample_size = min(1000, len(X_train_text))
train_texts = X_train_text.tolist()[:sample_size]
train_labels = y_train_text[:sample_size].reset_index(drop=True)

# Tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return bert_tokenizer(texts, truncation=True, padding=True, max_length=128)

train_encodings = tokenize(train_texts)

# Dataset class
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)

# Model
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()
# ================================
# EVALUATE DISTILBERT
# ================================
print("\nEvaluating DistilBERT...")

# Use small test sample (fast)
test_sample_size = min(1000, len(X_test_text))
test_texts = X_test_text.tolist()[:test_sample_size]
test_labels = y_test_text[:test_sample_size].reset_index(drop=True)

test_encodings = bert_tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_dataset = EmailDataset(test_encodings, list(test_labels))

predictions = trainer.predict(test_dataset)

bert_preds = np.argmax(predictions.predictions, axis=1)

from sklearn.metrics import accuracy_score
bert_acc = accuracy_score(test_labels, bert_preds)

print("DistilBERT Accuracy:", bert_acc)
# Save
bert_model.save_pretrained("models/email_bert_model")
bert_tokenizer.save_pretrained("models/email_bert_model")

print("DistilBERT training complete")

# ============================================
# 7. DEEP LEARNING MODELS (ADD HERE)
# ============================================

# Prepare text data
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    data['clean_text'], y, test_size=0.2, random_state=42, stratify=y
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout

MAX_WORDS = 10000
MAX_LEN = 200

# ---------------- CNN ----------------
print("\nTraining CNN...")

cnn_tokenizer = Tokenizer(num_words=MAX_WORDS)
cnn_tokenizer.fit_on_texts(X_train_text)

X_train_seq = cnn_tokenizer.texts_to_sequences(X_train_text)
X_test_seq = cnn_tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

cnn_model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(X_train_pad, y_train_text, epochs=2, batch_size=32)

cnn_pred = (cnn_model.predict(X_test_pad) > 0.5).astype(int)
print("CNN Accuracy:", accuracy_score(y_test_text, cnn_pred))

cnn_model.save("models/email_cnn_model.h5")
pickle.dump(cnn_tokenizer, open("models/email_cnn_tokenizer.pkl","wb"))

# ---------------- LSTM ----------------
print("\nTraining LSTM...")

lstm_tokenizer = Tokenizer(num_words=MAX_WORDS)
lstm_tokenizer.fit_on_texts(X_train_text)

X_train_seq = lstm_tokenizer.texts_to_sequences(X_train_text)
X_test_seq = lstm_tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

lstm_model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train_text, epochs=2, batch_size=32)

lstm_pred = (lstm_model.predict(X_test_pad) > 0.5).astype(int)
print("LSTM Accuracy:", accuracy_score(y_test_text, lstm_pred))

lstm_model.save("models/email_lstm_model.h5")
pickle.dump(lstm_tokenizer, open("models/email_lstm_tokenizer.pkl","wb"))

# ============================================
# 7. SAVE MODELS
# ============================================

os.makedirs("models", exist_ok=True)

with open("models/email_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/email_nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("models/email_lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/email_structured_cols.pkl", "wb") as f:
    pickle.dump(structured_features.columns.tolist(), f)

with open("models/email_phishing_keywords.pkl", "wb") as f:
    pickle.dump(PHISHING_KEYWORDS, f)

print("\nDONE! Models saved in 'models/' folder")