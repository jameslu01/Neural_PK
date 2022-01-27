import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(df, on_col, seed=2020, test_size=0.2):
    """Simple data split along a specified column."""
    target = df[on_col].unique()
    train, test = train_test_split(target, random_state=seed, test_size=test_size, shuffle=True)
    train_df = df[df[on_col].isin(train)]
    test_df = df[df[on_col].isin(test)]
    return train_df, test_df


def augment_data(train):
    """
    They add extra data to the training set made out of existing training data.
    Here is a description from the paper:

    "We applied augmentation to prevent overfitting.
    We applied timewise truncation to increase the number of training examples.
    For each training example, in addition to the original example, we also truncated
    the examples at 1008 hr, 1512 hr, and 2016 hr and generated and added
    a set of new examples to the training examples."
    """
    augment_data = pd.DataFrame(columns=train.columns)
    for ptnm in train.PTNM.unique():
        df = train[(train.PTNM == ptnm) & (train.TIME <= 2 * 21 * 24) & (train.TIME >= 0)]
        df.loc[:, "PTNM"] = df["PTNM"] + ".1"
        augment_data = pd.concat([augment_data, df], ignore_index=True)

        df = train[(train.PTNM == ptnm) & (train.TIME <= 3 * 21 * 24) & (train.TIME >= 0)]
        df.loc[:, "PTNM"] = df["PTNM"] + ".2"
        augment_data = pd.concat([augment_data, df], ignore_index=True)

        df = train[(train.PTNM == ptnm) & (train.TIME <= 4 * 21 * 24) & (train.TIME >= 0)]
        df.loc[:, "PTNM"] = df["PTNM"] + ".3"
        augment_data = pd.concat([augment_data, df], ignore_index=True)

    train = pd.concat([train, augment_data], ignore_index=True).reset_index(drop=True)
    return train
