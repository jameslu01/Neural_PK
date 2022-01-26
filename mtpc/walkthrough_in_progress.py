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
from data_split import data_split, augment_data


BASE_RANDOM_SEED = 1329
SPLIT_FRAC = 0.2
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
data = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]


"""
The authors of this code are doing two things when it comes to splitting the data:

(1) They are doing 5 train/test splits. They repeat the entire training and test procedure 5 times.
(2) Within each of the 5 splits above, the are also doing model averaging by training 5 Neural ODE models
that differ in: (a) initial conditions, (b) random seeds and (c) which subset of the training data is used
for actual model training vs validation. These 5 models are then averaged together to get the final model
which is then applied to the test set.
"""
for fold in [1, 2, 3, 4, 5]:
    for model in [1, 2, 3, 4, 5]:

        # first we split up the data into training/validation/test
        train, test = data_split(data, "PTNM", seed=BASE_RANDOM_SEED + fold, test_size=SPLIT_FRAC)
        train, validate = data_split(train, "PTNM", seed=BASE_RANDOM_SEED + fold + model, test_size=SPLIT_FRAC)

        """
        Adding the first cycle of treatment of the test set to the training set, as it will later be used
        during test to predict the rest of the test set and not for evaluation. As such, the authors
        think it is OK to add to training data and to maximize the total amount of training data.
        
        TODO: confirm that the test_add_to_train was NOT actually used in the final evaluation metrics. 
        
        Reasoning from the paper:
        "Additionally, the first cycle of the observation, PK_cycle1 is also available as predictive features for the models. 
        Using the information above, we sought to predict the PK dynamics after the first cycle, i.e., 
        after 168 hr for the Q1W data and after 504 hr for the Q3W data."
        """
        test_add_to_train = pd.concat(
            [test[(test.DSFQ == 1) & (test.TIME < 168)], test[(test.DSFQ == 3) & (test.TIME < 504)]], ignore_index=True
        )
        train = pd.concat([train, test_add_to_train], ignore_index=True)
        # i am not sure it makes sense to add this to the validation data?
        validate = pd.concat([validate, test_add_to_train], ignore_index=True)

        """
        They add extra data to the training set made out of existing training data. 
        Here is a description from the paper:

        "We applied augmentation to prevent overfitting.
        We applied timewise truncation to increase the number of training examples.
        For each training example, in addition to the original example, we also truncated
        the examples at 1008 hr, 1512 hr, and 2016 hr and generated and added
        a set of new examples to the training examples."
        """
        train = augment_data(train)

        # CUDA_VISIBLE_DEVICES="" python run_train.py --fold $fold --model $model --save fold_$fold --lr 0.00005 --tol 1e-4 --epochs 30 --l2 0.1
        # CUDA_VISIBLE_DEVICES="" python run_predict.py --fold $fold --model $model --save fold_$fold --tol 1e-4
