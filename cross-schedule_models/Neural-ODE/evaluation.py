
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


def score(df, label_col, pred_col, score_fn):
    y_true = df[label_col].values
    y_pred = df[pred_col].values
    if score_fn is mean_squared_error:
        return score_fn(y_true, y_pred, squared=False)
    elif score_fn is pearsonr:
        return score_fn(y_true, y_pred)[0]
    else:
        return score_fn(y_true, y_pred)
    

def merge_predictions(files, method="mean"):
    cols = ['PTNM','TIME','preds']
    left = pd.read_csv(files[0])
    for f in files[1:]:
        right = pd.read_csv(f)
        left = left.merge(right[cols], on=["PTNM", "TIME"], how="left")
    preds = [col for col in left.columns.values if col.startswith("preds")]
    left["pred_agg"] = left[preds].agg(method, axis=1)
    left = left[left.TIME >= 168]
    print(left.shape)
    return left


def write_score(pred_df, fold, model_name, rmse_dict, score_fn):
    res = score(pred_df, "labels", "pred_agg", score_fn)
    if not rmse_dict.get("fold", []) or rmse_dict["fold"][-1] != fold:
        rmse_dict["fold"] = rmse_dict.get("fold", []) + [fold]
    rmse_dict[model_name] = rmse_dict.get(model_name, []) + [round(res, 5)]


def main(score_type, score_fn, args):
    records = {}
    for fold in args.folds:
        in_file = ["fold_{}/fold_{}_model_{}.csv".format(fold, fold, m) for m in args.models]
        predictions = merge_predictions(in_file)
        predictions.to_csv("predictions.csv", index=False)
        write_score(predictions, fold, "ensemble (mean)", records, score_fn)

        for f in in_file:
            predictions = merge_predictions([f])
            write_score(predictions, fold, f.split("/")[1], records, score_fn)

    df = pd.DataFrame(records)
    print(df)
    summary_df = df.drop(columns="fold").agg(["min", "max", "mean", "median"])
    print(summary_df)
    df.to_csv(f"{score_type}.txt", sep="\t", index=False)
    summary_df.to_csv(f"{score_type}_summary.txt", sep="\t", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--models", type=int, nargs="+", required=True)
    args = parser.parse_args()

    main("rmse", mean_squared_error, args)
    main("r2", r2_score, args)
    main("correlation", pearsonr, args)
