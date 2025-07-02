# Task 1: Subjectivity in News Articles

Systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author behind it or presents an objective view on the covered topic instead.

This is a binary classification tasks in which systems have to identify whether a text sequence (a sentence or a paragraph) is subjective (**SUBJ**) or objective (**OBJ**).

The task comprises three settings:
- **Monolingual**: train and test on data in a given language L
- **Multilingual**: train and test on data comprising several languages
- **Zero-shot**: train on several languages and test on unseen languages



## Contents of the Directory

- Main folder: [data](./data)
  - Contains a subfolder for each language which contain the data as TSV format with .tsv extension (train_LANG.tsv, dev_LANG.tsv, dev_test_LANG.tsv, test_LANG.tsv).
  As LANG we used standard language code for each language.
- Main folder: [baseline](./baseline)<br/>
  - Contains a single file, baseline.py, used to train a baseline and provide predictions.
- Notebook folder<br/>
  - Contains all the notebook for our model related for each setting of all langiages.
- unlabeld_predict folder<br/>
  - Contains the predications for the languages.
- [README.md](./README.md) <br/>

## Datasets statistics

* **English**
  - train: 830 sentences, 532 OBJ, 298 SUBJ
  - dev: 462 sentences, 222 OBJ, 240 SUBJ
  - dev-test: 484 sentences, 362 OBJ, 122 SUBJ
  - test: TBA
* **Italian**
  - train: 1613 sentences, 1231 OBJ, 382 SUBJ
  - dev: 667 sentences, 490 OBJ, 177 SUBJ
  - dev-test - 513 sentences, 377 OBJ, 136 SUBJ
  - test: TBA
* **German**
  - train: 800 sentences, 492 OBJ, 308 SUBJ
  - dev: 491 sentences, 317 OBJ, 174 SUBJ
  - dev-test - 337 sentences, 226 OBJ, 111 SUBJ
  - test: TBA
* **Bulgarian**
  - train: 729 sentences, 406 OBJ, 323 SUBJ
  - dev: 467 sentences, 175 OBJ, 139 SUBJ
  - dev-test - 250 sentences, 143 OBJ, 107 SUBJ
  - test: TBA
* **Arabic**
  - train: 2,446 sentences, 1391 OBJ, 1055 SUBJ
  - dev: 742 sentences, 266 OBJ, 201 SUBJ
  - dev-test - 748 sentences, 425 OBJ, 323 SUBJ
  - test: TBA

## Input Data Format

The data will be provided as a TSV file with three columns:
> sentence_id <TAB> sentence <TAB> label

Where: <br>
* sentence_id: sentence id for a given sentence in a news article<br/>
* sentence: sentence's text <br/>
* label: *OBJ* and *SUBJ*

<!-- **Note:** For English, the training and development (validation) sets will also include a fourth column, "solved_conflict", whose boolean value reflects whether the annotators had a strong disagreement. -->

**Examples:**

> b9e1635a-72aa-467f-86d6-f56ef09f62c3  Gone are the days when they led the world in recession-busting SUBJ
>
> f99b5143-70d2-494a-a2f5-c68f10d09d0a  The trend is expected to reverse as soon as next month.  OBJ

## Output Data Format

The output must be a TSV format with two columns: sentence_id and label.

## Evaluation Metrics

This task is evaluated as a classification task. We will use the F1-macro measure for the ranking of teams.

We will also measure Precision, Recall, and F1 of the SUBJ class and the macro-averaged scores.
<!--
There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt -->


## Baselines

The script to train the baseline is provided in the related directory.
The script can be run as follow:

> python baseline.py -trp train_data.tsv -ttp dev_data.tsv

where train_data.tsv is the file to be used for training and dev_data.tsv is the file on which doing the prediction.

The baseline is a logistic regressor trained on a Sentence-BERT multilingual representation of the data.

<!-- ### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498
 -->

## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task1/
