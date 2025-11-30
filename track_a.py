import numpy as np
import pandas as pd
import nltk


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer, util

# BM25
from rank_bm25 import BM25Okapi

MODEL_NAMES = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
print("Loading SBERT models...")
sbert_models = [SentenceTransformer(name) for name in MODEL_NAMES]

tfidf = TfidfVectorizer()

def get_pos_words(text, prefix):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return {word.lower() for word, tag in tagged if tag.startswith(prefix)}

def compute_features(anchor, a, b):
    sims_A = []
    sims_B = []

    for model in sbert_models:
        emb = model.encode([anchor, a, b], convert_to_tensor=True)
        anchor_emb, a_emb, b_emb = emb

        sim_a = util.cos_sim(anchor_emb, a_emb).item()
        sim_b = util.cos_sim(anchor_emb, b_emb).item()
        sims_A.append(sim_a)
        sims_B.append(sim_b)

    len_anchor = len(anchor.split())
    len_a = len(a.split())
    len_b = len(b.split())

    nouns_anchor = get_pos_words(anchor, "NN")
    nouns_a = get_pos_words(a, "NN")
    nouns_b = get_pos_words(b, "NN")

    verbs_anchor = get_pos_words(anchor, "VB")
    verbs_a = get_pos_words(a, "VB")
    verbs_b = get_pos_words(b, "VB")

    noun_overlap_a = len(nouns_anchor & nouns_a)
    noun_overlap_b = len(nouns_anchor & nouns_b)
    verb_overlap_a = len(verbs_anchor & verbs_a)
    verb_overlap_b = len(verbs_anchor & verbs_b)

    tfidf_matrix = tfidf.fit_transform([anchor, a, b])
    tfidf_sim_a = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    tfidf_sim_b = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])[0][0]

    tokenized_anchor = anchor.lower().split()
    tokenized_a = a.lower().split()
    tokenized_b = b.lower().split()

    bm25 = BM25Okapi([tokenized_anchor])

    bm25_a = bm25.get_scores(tokenized_a)[0]
    bm25_b = bm25.get_scores(tokenized_b)[0]
    bm25_diff = bm25_a - bm25_b

    return [
        np.mean(sims_A),
        np.mean(sims_B),
        np.mean(sims_A) - np.mean(sims_B),

        len_anchor,
        len_a,
        len_b,
        abs(len_anchor - len_a),
        abs(len_anchor - len_b),

        noun_overlap_a,
        noun_overlap_b,
        verb_overlap_a,
        verb_overlap_b,

        tfidf_sim_a,
        tfidf_sim_b,
        tfidf_sim_a - tfidf_sim_b,

        bm25_a,
        bm25_b,
        bm25_diff
    ]

print("Loading dev_track_a.jsonl...")
df = pd.read_json("data/dev_track_a.jsonl", lines=True)

# Feature matrix:
print("Generating full feature matrix...")
X = []
y = []

for _, row in df.iterrows():
    feats = compute_features(row["anchor_text"], row["text_a"], row["text_b"])
    X.append(feats)
    y.append(bool(row["text_a_is_closer"]))

X = np.array(X)
y = np.array(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = []
stack_train = np.zeros((len(X), 5))  # meta-model inputs

fold_idx = 0

print("\nTraining 5 XGBoost models (one per fold)...\n")

for train_idx, val_idx in kf.split(X):
    fold_idx += 1
    print(f"Training fold {fold_idx}...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=350,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss"
    )

    clf.fit(X_train, y_train)
    models.append(clf)

    preds = clf.predict_proba(X_val)[:, 1]
    stack_train[val_idx, fold_idx - 1] = preds

    acc = accuracy_score(y_val, preds >= 0.5)
    print(f"Fold {fold_idx} accuracy: {acc:.3f}")

print("\nTraining Logistic Regression meta-classifier...")
meta_clf = LogisticRegression(max_iter=500)
meta_clf.fit(stack_train, y)

print("\nPredicting on full dataset...")

stack_test = np.zeros((len(X), 5))
for i, clf in enumerate(models):
    stack_test[:, i] = clf.predict_proba(X)[:, 1]

final_preds = meta_clf.predict(stack_test)

overall_acc = accuracy_score(y, final_preds)
print(f"\nFinal ensemble accuracy: {overall_acc:.3f}")

df["text_a_is_closer"] = final_preds.astype(bool)
df.to_json("output/track_a.jsonl", orient="records", lines=True, force_ascii=False)

print("\nSaved → output/track_a.jsonl\n")
