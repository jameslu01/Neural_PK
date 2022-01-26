"""
Here is run.sh, that we will reproduce here
```
python process_data.py
for fold in 1 2 3 4 5; do
    for model in 1 2 3 4 5; do
        python data_split.py --data data.csv --fold $fold --model $model

        CUDA_VISIBLE_DEVICES="" python run_train.py --fold $fold --model $model --save fold_$fold --lr 0.00005 --tol 1e-4 --epochs 30 --l2 0.1
        CUDA_VISIBLE_DEVICES="" python run_predict.py --fold $fold --model $model --save fold_$fold --tol 1e-4
    done
done    
```
Note: evaluation.py is never called in run.sh but we will add it in properly.
"""

import pandas as pd


"""
Example data has the following columns:
  - STUD - Study ID. Can be 1000, 2000, 3000.
  - PTNM - Patient number. Can be repeated between studies, but for example patient 1 in Study 1000 is not the same person as patient 1 in study 2000.
  - DSFQ - Dosage frequency is how often the dose is administred. Only 1 or 3.
  - AMT - Dosage amount. Can be 0 when measurements taken between doses.
  - TIME - Time since beginning of patient's treatment.
  - TFDS - Time since dose.
  - DV - Concentration measurement. 
  
# TODO: comment the code below
"""


data_complete = pd.read_csv("ExampleData/sim_data.csv", na_values=".")

select_cols = ["STUD", "DSFQ", "PTNM", "CYCL", "AMT", "TIME", "TFDS", "DV"]
if "C" in data_complete.columns.values:
    data_complete = data_complete[data_complete.C.isnull()]
data_complete = data_complete[data_complete.CYCL < 100]
data_complete = data_complete[select_cols]
data_complete = data_complete.rename(columns={"DV": "PK_timeCourse"})
data_complete["PTNM"] = data_complete["PTNM"].astype("int").map("{:05d}".format)
data_complete["ID"] = data_complete["STUD"].astype("int").astype("str") + data_complete["PTNM"]

time_summary = data_complete[["ID", "TIME"]].groupby("ID").max().reset_index()
selected_ptnms = time_summary[time_summary.TIME > 0].ID
data_complete = data_complete[data_complete.ID.isin(selected_ptnms)]

data_complete["AMT"] = data_complete["AMT"].fillna(0)
data_complete["PK_round1"] = data_complete["PK_timeCourse"]
data_complete.loc[(data_complete.DSFQ == 1) & (data_complete.TIME >= 168), "PK_round1"] = 0
data_complete.loc[(data_complete.DSFQ == 3) & (data_complete.TIME >= 504), "PK_round1"] = 0
data_complete["PK_round1"] = data_complete["PK_round1"].fillna(0)
data_complete["PK_timeCourse"] = data_complete["PK_timeCourse"].fillna(-1)

data_complete = data_complete[~((data_complete.AMT == 0) & (data_complete.TIME == 0))]
data_complete.loc[data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"] = data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"
].values
data_complete = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]

data_complete.to_csv("data.csv", index=False)
