import csv

from typing import AnyStr
import argparse
import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging


def validate_files(
        pred_file: AnyStr,
        gold_file: AnyStr
):
    pred_data = pandas.read_csv(pred_file, sep='\t', quoting=csv.QUOTE_NONE)
    gold_data = pandas.read_csv(gold_file, sep='\t', quoting=csv.QUOTE_NONE)

    if len(pred_data) != len(gold_data):
        logging.error("Error! Different number of instances in the files")
        return False

    if not ('label' in pred_data and
            'sentence_id' in pred_data and 'sentence_id' in gold_data):
        logging.error("Error! Wrong columns")
        return False

    pred_values = pred_data['label'].unique()

    if not ((len(pred_values) == 2 and "SUBJ" in pred_values and
             "OBJ" in pred_values) or
            (len(pred_values) == 1 and
             ("OBJ" in pred_values or "SUBJ" in pred_values))):
        logging.error("Error! Wrong labels")
        return False

    pred_data.rename(columns={'label': 'pred_label'}, inplace=True)

    whole_data = pandas.merge(pred_data, gold_data, on="sentence_id")

    if len(pred_data) != len(whole_data):
        logging.error("Error! Different ids in the two files")
        return False

    if not ('label' in gold_data):
        logging.error("Error! No labels in the gold data file. Impossible to proceed with evaluation")
        return False

    logging.info("The file is properly formatted")
    whole_data.rename(columns={'label': 'gold_label'}, inplace=True)

    return whole_data


def evaluate(
        whole_data
):
    pred_values = whole_data['pred_label'].values
    gold_values = whole_data['gold_label'].values

    acc = accuracy_score(gold_values, pred_values)
    m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(gold_values, pred_values, average="macro",
                                                               zero_division=0)
    p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(gold_values, pred_values, labels=["SUBJ"],
                                                               zero_division=0)

    return {
        'macro-F1': m_f1,
        'macro-P': m_prec,
        'macro-R': m_rec,
        'SUBJ-F1': p_f1[0],
        'SUBJ-P': p_prec[0],
        'SUBJ-R': p_rec[0],
        'accuracy': acc
    }


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to file with predict class per sentence")

    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    whole_data = validate_files(pred_file, gold_file)

    if whole_data is not False:
        logging.info("Started evaluating results for Task 1...")

        scores = evaluate(whole_data)

        logging.info(f"""
            macro-F1: {scores['macro-F1']:.4f}
            macro-P: {scores['macro-P']:.4f}
            macro-R: {scores['macro-R']:.4f}
            
            SUBJ-F1: {scores['SUBJ-F1']:.4f}
            SUBJ-P: {scores['SUBJ-P']:.4f}
            SUBJ-R: {scores['SUBJ-R']:.4f}
            
            accuracy: {scores['accuracy']:.4f}
        """)
