import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

import utils
from model import Encoder, ODEFunc, Classifier
from data_parse import parse_tdm1


def train_neural_ode(
    random_seed, train, validate, test, model, fold, lr, tol, epochs, l2, hidden_dim, latent_dim, ode_hidden_dim
):
    # choose whether to use a GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set various seeds for complete reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # the model checkpoints will be stored in this directory
    utils.makedirs(f"fold_{fold}")
    ckpt_path = os.path.join(f"fold_{fold}", f"fold_{fold}_model_{model}.ckpt")

    # for the logging we'll have this informative string
    input_cmd = f"--fold {fold} --model {model} --lr {lr} --tol {tol} --epochs {epochs} --l2 {l2} --hidden_dim {hidden_dim} --laten_dim {latent_dim}"

    tdm1_obj = parse_tdm1(device, train, validate, test, phase="train")
    input_dim = tdm1_obj["input_dim"]

    # put the model together
    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=ode_hidden_dim)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

    # make the logs
    log_path = "logs/" + f"fold_{fold}_model_{model}.log"
    utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path)
    logger.info(input_cmd)

    batches_per_epoch = tdm1_obj["n_train_batches"]
    criterion = nn.MSELoss().to(device=device)
    params = list(encoder.parameters()) + list(ode_func.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
    best_rmse = 0x7FFFFFFF
    best_epochs = 0

    for epoch in range(1, epochs + 1):

        for _ in tqdm(range(batches_per_epoch), ascii=True):
            optimizer.zero_grad()

            ptnms, times, features, labels, cmax_time = tdm1_obj["train_dataloader"].__next__()
            dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
            dosing[:, :, 0] = features[:, :, -2]
            dosing = dosing.permute(1, 0, 2)

            encoder_out = encoder(features)
            qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
            z0 = utils.sample_standard_gaussian(qz0_mean, qz0_var)

            solves = z0.unsqueeze(0).clone()
            try:
                for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                    z0 += dosing[idx]
                    time_interval = torch.Tensor([time0 - time0, time1 - time0])
                    sol = odeint(ode_func, z0, time_interval, rtol=tol, atol=tol)
                    z0 = sol[-1].clone()
                    solves = torch.cat([solves, sol[-1:, :]], 0)
            except AssertionError:
                print(times)
                print(time0, time1, time_interval, ptnms)
                continue

            preds = classifier(solves, cmax_time)

            loss = utils.compute_loss_on_train(criterion, labels, preds)
            try:
                loss.backward()
            except RuntimeError:
                print(ptnms)
                print(times)
                continue
            optimizer.step()

        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        preds = preds.permute(1, 0, 2)[idx_not_nan]
        labels = labels[idx_not_nan]
        print(preds)
        print(labels)

        with torch.no_grad():

            train_res = utils.compute_loss_on_test(
                encoder,
                ode_func,
                classifier,
                tol,
                tdm1_obj["train_dataloader"],
                tdm1_obj["n_train_batches"],
                device,
                phase="train",
            )

            validation_res = utils.compute_loss_on_test(
                encoder,
                ode_func,
                classifier,
                tol,
                tdm1_obj["val_dataloader"],
                tdm1_obj["n_val_batches"],
                device,
                phase="validate",
            )

            train_loss = train_res["loss"]
            validation_loss = validation_res["loss"]
            if validation_loss < best_rmse:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "ode": ode_func.state_dict(),
                        "classifier": classifier.state_dict(),
                    },
                    ckpt_path,
                )
                best_rmse = validation_loss
                best_epochs = epoch

            message = """
            Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
            Best loss {:.6f} | Best epoch {:04d}
            """.format(
                epoch, train_loss, train_res["r2"], validation_loss, validation_res["r2"], best_rmse, best_epochs
            )
            logger.info(message)