import os
import pickle
from collections import Counter, defaultdict
from glob import glob
from itertools import chain, combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def powerset(iterable):
    "powerset([1,2,3]) --> [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


def get_id(path: str) -> str:
    """Convert file path into an id."""
    return (
        os.path.basename(path)
        .replace(".tsv", "")
        .replace("_train_", "")
        .replace("_dev_", "")
    )


def load(s: str) -> np.ndarray:
    """Parse array string into a numpy vector."""
    return np.array(s[1:-1].split(), dtype=np.float)


def load_dataframe(paths: List[str]):
    d: Dict[int, Dict[str, str]] = defaultdict(dict)
    for file in tqdm(paths):
        identifier = get_id(file)
        df: pd.DataFrame = pd.read_csv(file, sep="\t")
        df = df.sort_values(by="Sentence Index")
        for _, row in df.iterrows():
            d[row["Sentence Index"]][identifier] = row["Predictions"]
            d[row["Sentence Index"]]["y"] = row["Labels"]
    return pd.DataFrame(d).T


def load_dataset_logits(paths: List[str]):
    all_feats = []
    all_labels = []
    dfs = []
    sentence_ids = set()

    for file in tqdm(paths):
        df: pd.DataFrame = pd.read_csv(file, sep="\t")
        df = df.sort_values(by="Sentence Index")
        dfs.append(df)
        sentence_ids.update(df["Sentence Index"].tolist())

    for sid in tqdm(sentence_ids):
        sentence_feats = []
        for df in dfs:
            sentence_feats.append(load(df.iloc[sid]["SoftMaxes"]))

        sentence_label = dfs[0].iloc[sid]["Labels"]

        all_feats.append(np.concatenate(sentence_feats))
        all_labels.append(sentence_label)

    return all_feats, all_labels


def load_dataset_predictions(paths: List[str]):
    all_feats = []
    all_labels = []
    dfs = []
    sentence_ids = set()

    for file in tqdm(paths):
        df: pd.DataFrame = pd.read_csv(file, sep="\t")
        df = df.sort_values(by="Sentence Index")
        dfs.append(df)
        sentence_ids.update(df["Sentence Index"].tolist())

    for sid in tqdm(sentence_ids):
        sentence_feats = []
        for df in dfs:
            sentence_feats.append(df.iloc[sid]["Predictions"])

        sentence_label = dfs[0].iloc[sid]["Labels"]

        all_feats.append(sentence_feats)
        all_labels.append(sentence_label)

    print(len(all_feats), len(all_labels))

    return all_feats, all_labels


def majority(x):
    result = []
    for row in x:
        m = Counter(row).most_common(1)[0][0]
        result.append(m)
    return result


def compute_scores(y_true: List[str], y_pred: List[str], models: List[str]):
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    if f1_macro > 0.325:
        print(f"Models  : {list(models)}")
        print(f"Accuracy: {acc}")
        print(f"F1 Micro: {f1_micro}")
        print(f"F1 Macro: {f1_macro}")
        print()


def main():
    train_paths = sorted(
        glob("D:/Downloads/Train Dumps/Train Dumps/predictions_train_*.tsv")
    )
    valid_paths = sorted(
        glob(f"D:/Downloads/Dev_Dumps/Dev Dumps/predictions_dev_*.tsv")
    )

    df_train = load_dataframe(train_paths)
    df_valid = load_dataframe(valid_paths)

    print("Majority")

    for c in tqdm(powerset(list(map(get_id, valid_paths)))):
        y_pred = majority(df_valid[list(c)].values)
        y_true = df_valid["y"].values
        compute_scores(y_true, y_pred, list(c))

    print("Model")

    for c in tqdm(powerset(list(map(get_id, train_paths)))):
        model = RandomForestClassifier()
        X_train = df_train[list(c)].values
        y_train = df_train["y"].values
        X_valid = df_valid[list(c)].values
        y_true = df_valid["y"].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        compute_scores(y_true, y_pred, list(c))


if __name__ == "__main__":
    main()
