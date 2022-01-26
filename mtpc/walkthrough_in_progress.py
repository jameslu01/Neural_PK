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
# According to authors: Patient data that have been marked with non-missing values in the "C" columns have been removed from the analysis
if "C" in data_complete.columns.values:
    data_complete = data_complete[data_complete.C.isnull()]
data_complete = data_complete[data_complete.CYCL < 100]  # cut off all dosing cycles greater than 100
data_complete = data_complete[select_cols]  # filter down to columns of interest
data_complete = data_complete.rename(
    columns={"DV": "PK_timeCourse"}
)  # DV is our variable of interest - anolyte concentration
data_complete["PTNM"] = data_complete["PTNM"].astype("int").map("{:05d}".format)
data_complete["ID"] = (
    data_complete["STUD"].astype("int").astype("str") + data_complete["PTNM"]
)  # concatenate study ID and patient ID for overall, unique ID

time_summary = (
    data_complete[["ID", "TIME"]].groupby("ID").max().reset_index()
)  # get max time since start of treatment per ID
# only keep patients who have measurements past initial measurements (TIME == 0)
selected_ptnms = time_summary[time_summary.TIME > 0].ID
data_complete = data_complete[data_complete.ID.isin(selected_ptnms)]

data_complete["AMT"] = data_complete["AMT"].fillna(0)  # replace missing values for dosage with 0s
# Set up round 1 measurement features. round 1 for each ID is used as intput for prediction of measurements that occur after round 1.
# For weekly dosage IDs, cut off after end of week 1 (TIME == 168), for every 3 week dosage IDs, cut off after end of week 3 (TIME == 604)
data_complete["PK_round1"] = data_complete["PK_timeCourse"]
data_complete.loc[(data_complete.DSFQ == 1) & (data_complete.TIME >= 168), "PK_round1"] = 0
data_complete.loc[(data_complete.DSFQ == 3) & (data_complete.TIME >= 504), "PK_round1"] = 0

# Missing PK measurement value handling
# TODO: Check with authors why they chose to fill these values differently
data_complete["PK_round1"] = data_complete["PK_round1"].fillna(0)  # round 1 missing values filled with 0
data_complete["PK_timeCourse"] = data_complete["PK_timeCourse"].fillna(-1)  # all others filled with -1

data_complete = data_complete[
    ~((data_complete.AMT == 0) & (data_complete.TIME == 0))
]  # drop all first patient rows with no dosage

# some rows are duplicate pairs for PTNM and TIME combinations
# set the first dosage amount of duplicated rows to the last dosage amount and keep only first row of duplicated rows
# This implementation may be an issue if patient number (PTNM) repeats across multiple studies (STUD)
data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"  # all non-last duplicated rows
] = data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"  # all non-first duplicated rows
].values
data = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]
