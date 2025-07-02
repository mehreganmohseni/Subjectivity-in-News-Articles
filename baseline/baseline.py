import argparse
import csv
import logging
import random
from pathlib import Path
from typing import AnyStr

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

random.seed(0)


def sanitize_and_check_filepath(
        filepath: AnyStr
) -> Path:
    filepath = Path(filepath).resolve()

    if not filepath.exists() or not filepath.is_file():
        raise RuntimeError(f'Could not find file. Got {filepath}')

    return filepath


def run_sbert_lr_baseline(
        data_dir: Path,
        train_filepath: Path,
        test_filepath: Path
) -> Path:
    train_data = pd.read_csv(train_filepath, sep='\t', quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv(test_filepath, sep='\t', quoting=csv.QUOTE_NONE)

    vect = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    model = LogisticRegression(class_weight="balanced")
    model.fit(X=vect.encode(train_data['sentence'].values), y=train_data['label'].values)

    predictions = model.predict(X=vect.encode(test_data['sentence'].values)).tolist()
    pred_df = pd.DataFrame()
    pred_df['sentence_id'] = test_data['sentence_id']
    pred_df['label'] = predictions

    predictions_filepath = data_dir.joinpath('baseline_pred.tsv')
    pred_df.to_csv(predictions_filepath, index=False, sep='\t', quoting=csv.QUOTE_NONE)

    return predictions_filepath


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', '-trp',
                        required=True,
                        type=str, )
    parser.add_argument('--testpath', '-ttp',
                        required=True,
                        type=str, )
    args = parser.parse_args()

    train_filepath = sanitize_and_check_filepath(args.trainpath)
    test_filepath = sanitize_and_check_filepath(args.testpath)

    data_dir = test_filepath.parent

    logging.info(f"""Running baseline with following configuration: 
                 Train: {train_filepath} 
                 Test: {test_filepath}""")
    run_sbert_lr_baseline(data_dir=data_dir,
                          test_filepath=test_filepath,
                          train_filepath=train_filepath)
