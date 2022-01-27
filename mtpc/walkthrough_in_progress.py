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
from train_predict_utils import train_neural_ode, predict_using_trained_model

# general hyperparameters
BASE_RANDOM_SEED = 1329
TORCH_RANDOM_SEED = 1000  # they have different random seeds for splitting and for the neural network
SPLIT_FRAC = 0.2

# hyperparemeters for the model, selected by grid search
# TODO: describe what each one does
# NOTE: the authors don't discuss how they chose these hyperparameters
LR = 0.00005
TOL = 1e-4
EPOCHS = 30
L2 = 0.1
HIDDEN_DIM = 128
LATENT_DIM = 6
HIDDEN_DIM = 128
ODE_HIDDEN_DIM = 16

"""
Example data has the following columns:
  - STUD - Study ID. Can be 1000, 2000, 3000.
  - PTNM - Patient number. Can be repeated between studies, but for example patient 1 in Study 1000 is not the same person as patient 1 in study 2000.
  - DSFQ - Dosage frequency is how often the dose is administred. Only 1 or 3.
  - AMT - Dosage amount. Can be 0 when measurements taken between doses.
  - TIME - Time since beginning of patient's treatment.
  - TFDS - Time since dose.
  - DV - Concentration measurement. 
  
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

# Set up round 1 measurement features.
# Round 1 measurements for each ID are always used as input features for the neural network to predict measurements after round 1.
# For weekly dosage IDs, round 1 is anything before end of week 1 (TIME <= 168), for every 3 week dosage IDs, anything before end of week 3 (TIME <= 604)
data_complete["PK_round1"] = data_complete["PK_timeCourse"]
data_complete.loc[(data_complete.DSFQ == 1) & (data_complete.TIME >= 168), "PK_round1"] = 0
data_complete.loc[(data_complete.DSFQ == 3) & (data_complete.TIME >= 504), "PK_round1"] = 0

# Missing PK measurement value handling
data_complete["PK_round1"] = data_complete["PK_round1"].fillna(0)  # round 1 missing values filled with 0
data_complete["PK_timeCourse"] = data_complete["PK_timeCourse"].fillna(
    -1
)  # all others filled with -1, used to find missing values during training

data_complete = data_complete[
    ~((data_complete.AMT == 0) & (data_complete.TIME == 0))
]  # drop all first patient rows with no dosage

# Some rows are duplicate pairs for PTNM and TIME combinations with different cycle (CYCL) values
# Set the first dosage amount of duplicated rows to the last dosage amount and keep only last row of duplicated rows
# This implementation may be an issue if patient number (PTNM) repeats across multiple studies (STUD)
data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"  # all non-last duplicated rows
] = data_complete.loc[
    data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"  # all non-first duplicated rows
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

        # create and train the model
        # the best checkpoint will be saved
        train_neural_ode(
            TORCH_RANDOM_SEED + model + fold,
            train,
            validate,
            model,
            fold,
            LR,
            TOL,
            EPOCHS,
            L2,
            HIDDEN_DIM,
            LATENT_DIM,
            ODE_HIDDEN_DIM,
        )

        # predict on test using the best model saved
        # during train_neural_ode
        predict_using_trained_model(
            test,
            model,
            fold,
            TOL,
            HIDDEN_DIM,
            LATENT_DIM,
            ODE_HIDDEN_DIM,
        )

        # TODO(anyone):
        # implement the evaluation from evaluation.py
        # go through and see which utils are never used and delete them
        # add a lot of documentation everywhere
        # convert to jupyter notebook
        # go back through emails and make sure we've answered all questions
