
import os
import numpy as np
import pandas as pd

import torch

import utils
from args import args
from model import *
from data_parse import parse_tdm1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################
## Main runnings
ckpt_path = os.path.join(args.save, f"fold_{args.fold}_model_{args.model}.ckpt")
eval_path = os.path.join(args.save, f"fold_{args.fold}_model_{args.model}.csv")
res_path = "rmse.csv"

########################################################################
tdm1_obj = parse_tdm1(device, phase="test")
input_dim = tdm1_obj["input_dim"]
hidden_dim = 128 
latent_dim = 6

encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=16)
classifier = Classifier(latent_dim=latent_dim, output_dim=1)

utils.load_model(ckpt_path, encoder, ode_func, classifier, device)

########################################################################
## Predict & Evaluate
with torch.no_grad():
    test_res = utils.compute_loss_on_test(encoder, ode_func, classifier, args,
        tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
        device, phase="test")

eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path, index=False)

"""
with torch.no_grad():
    test_res = utils.compute_loss_on_interp(encoder, ode_func, classifier, args,
        tdm1_obj["interp_dataloader"], tdm1_obj["test_dataloader"], tdm1_obj["n_interp_batches"], 
        device, phase="test")

eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path + ".interp", index=False)

with torch.no_grad():
    test_res = utils.compute_loss_on_interp(encoder, ode_func, classifier, args,
        tdm1_obj["nodosing_dataloader"], tdm1_obj["test_dataloader"], tdm1_obj["n_interp_batches"], 
        device, phase="test")

eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path + ".nodosing", index=False)
"""



