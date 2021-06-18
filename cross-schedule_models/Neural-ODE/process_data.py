
## not for running, just for showing how to get the data from the original
## start from TDM1_LiverToxData_corrected_17Feb2020.csv

import pandas as pd

data_complete = pd.read_csv("../../ExampleData/sim_data.csv", na_values='.')

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
data_complete.loc[data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"] = \
    data_complete.loc[data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"].values
data_complete = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]

data_complete.to_csv("data.csv", index=False)
