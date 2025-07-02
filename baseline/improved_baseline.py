#!/usr/bin/env python3
"""
baseline.py

Enhanced subjectivity detection merging SBERT embeddings,
TF–IDF n-grams, and lexicon & surface cues, with optional hyperparameter tuning.
Parses MPQA .tff lexicon format directly.
"""
import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def sanitize_and_check_filepath(filepath: str) -> Path:
    path = Path(filepath).resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"File not found: {path}")
    return path


def load_mpqa_tff(lex_path: Path):
    pos_lex = set()
    neg_lex = set()
    with open(lex_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            attrs = {}
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    attrs[k] = v
            word = attrs.get('word1')
            prior = attrs.get('priorpolarity')
            if not word or not prior:
                continue
            if prior == 'positive':
                pos_lex.add(word)
            elif prior == 'negative':
                neg_lex.add(word)
    return pos_lex, neg_lex


def run_baseline(data_dir: Path,
                 train_filepath: Path,
                 test_filepath: Path,
                 lexicon_path: Path = None,
                 tune: bool = False) -> Path:
    """
    Load data, extract features (SBERT, TF–IDF, lexicon/surface cues),
    train LogisticRegression (with optional tuning), and predict on test.
    """
    # Load data
    train_df = pd.read_csv(train_filepath, sep='\t', quoting=csv.QUOTE_NONE)
    test_df  = pd.read_csv(test_filepath,  sep='\t', quoting=csv.QUOTE_NONE)

    # Preprocess
    for df in (train_df, test_df):
        df['sentence'] = df['sentence'].str.lower().str.strip()

    # Load MPQA lexicon if provided
    pos_lex, neg_lex = set(), set()
    unc_lex = set()
    if lexicon_path:
        lex_path = sanitize_and_check_filepath(str(lexicon_path))
        pos_lex, neg_lex = load_mpqa_tff(lex_path)

    # Feature extractor combining lexicon & surface cues
    def lexsurf(texts):
        feats = []
        for text in texts:
            tokens = text.split()
            pos_cnt = sum(t in pos_lex for t in tokens)
            neg_cnt = sum(t in neg_lex for t in tokens)
            unc_cnt = sum(t in unc_lex for t in tokens)
            feats.append([
                pos_cnt,
                neg_cnt,
                unc_cnt,
                text.count('!'),
                text.count('?'),
                sum(w.isupper() for w in tokens) / len(tokens) if tokens else 0.0
            ])
        return np.array(feats)

    # Prepare transformers
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    sbert_trans   = FunctionTransformer(lambda texts: sbert.encode(texts, show_progress_bar=True), validate=False)
    lexsurf_trans = FunctionTransformer(lambda texts: lexsurf(texts),                    validate=False)

    features = FeatureUnion([
        ('tfidf',   tfidf),
        ('sbert',   sbert_trans),
        ('lexsurf', lexsurf_trans)
    ])

    pipeline = Pipeline([
        ('features', features),
        ('scaler',   StandardScaler(with_mean=False)),
        ('clf',      LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    X_train = train_df['sentence'].tolist()
    y_train = train_df['label'].tolist()
    X_test  = test_df['sentence'].tolist()

    if tune:
        param_dist = {
            'clf__C': [0.01, 0.1, 1, 10],
            'features__tfidf__max_features': [2000, 5000, 10000]
        }
        search = RandomizedSearchCV(
            pipeline, param_dist, cv=5,
            scoring='f1_macro', n_iter=10,
            verbose=2, n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logging.info(f"Best params: {search.best_params_}")
    else:
        model = pipeline.fit(X_train, y_train)

    preds = model.predict(X_test)
    out_df = pd.DataFrame({
        'sentence_id': test_df['sentence_id'],
        'label': preds
    })
    out_path = data_dir / 'baseline_pred.tsv'
    out_df.to_csv(out_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    return out_path


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath',  '-tr', required=True)
    parser.add_argument('--testpath',   '-ts', required=True)
    parser.add_argument('--outdir',     '-o',  required=True)
    parser.add_argument('--lexiconpath','-lx', help='MPQA TFF lexicon path')
    parser.add_argument('--tune',       '-t',  action='store_true')
    args = parser.parse_args()

    train_fp = sanitize_and_check_filepath(args.trainpath)
    test_fp  = sanitize_and_check_filepath(args.testpath)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Running baseline:\n Train: {train_fp}\n Test: {test_fp}")
    result_path = run_baseline(
        data_dir=outdir,
        train_filepath=train_fp,
        test_filepath=test_fp,
        lexicon_path=args.lexiconpath,
        tune=args.tune
    )
    logging.info(f"Predictions saved to {result_path}")
