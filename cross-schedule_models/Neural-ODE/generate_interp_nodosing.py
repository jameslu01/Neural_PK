
import pandas as pd
import numpy as np

def generate_interp(ptnm):
    test = pd.read_csv("test.csv")
    subset = test[test.PTNM == ptnm]
    interp_times = pd.DataFrame({"TIME": np.linspace(subset.loc[subset.TIME >= 168, "TIME"].values[:-1], \
                             subset.loc[subset.TIME >= 168, "TIME"].values[1:], 6, endpoint=False).flatten("F")})
    subset = pd.merge(subset, interp_times, how="outer", on="TIME").sort_values(by="TIME").reset_index(drop=True)
    subset["AMT"] = subset["AMT"].fillna(0)
    subset["PK_round1"] = subset["PK_round1"].fillna(0)
    subset["PTNM"] = subset["PTNM"].fillna(method="ffill")
    subset["DSFQ"] = subset["DSFQ"].fillna(1)
    subset["CYCL"] = subset["CYCL"].fillna(method="ffill")
    subset["PK_timeCourse"] = subset["PK_timeCourse"].fillna(-10) 
    
    start_time = 0
    for idx, feature in subset.iterrows():

        if feature["AMT"] > 0:
            start_time = feature["TIME"]
            continue

        subset.loc[idx, "TFDS"] = subset.loc[idx, "TIME"] - start_time

    return subset.drop(columns=["STUD", "ID"])

ptnms = ['5010','5011','5012','5013','5014','5015','5016','5209','5210','5211','5212','5409','5410','5411','5412','5413','5414','5415','5416','5602','5603','5604','5605','5606','5607','5608','5609','5610']
test_interp = pd.DataFrame()
for ptnm in ptnms:
    print(ptnm)
    test_interp = pd.concat([test_interp, generate_interp(int(ptnm))], ignore_index=True)
test_interp.to_csv("test_interp.csv", index=False)

test = pd.read_csv("test.csv")
test.loc[test.CYCL > 5, "AMT"] = 0
test.to_csv("test_nodosing.csv", index=False)
