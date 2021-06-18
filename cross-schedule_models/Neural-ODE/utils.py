import os
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from torchdiffeq import odeint_adjoint as odeint


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def load_model(ckpt_path, encoder=None, ode_func=None, classifier=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    checkpt = torch.load(ckpt_path)
    if encoder is not None:
        encoder_state = checkpt["encoder"]
        encoder.load_state_dict(encoder_state)
        encoder.to(device)

    if ode_func is not None:
        ode_state = checkpt["ode"]
        ode_func.load_state_dict(ode_state)
        ode_func.to(device)

    if classifier is not None:
        classifier_state = checkpt["classifier"]
        classifier.load_state_dict(classifier_state)
        classifier.to(device)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    """
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    """

    return logger


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def reverse(tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    return tensor[idx]


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)
    d = torch.distributions.normal.Normal(
            torch.Tensor([0.]).to(device),
            torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def compute_loss_on_train(criterion, labels, preds):
    preds = preds.permute(1, 0, 2)

    idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
    # print(idx_not_nan)
    preds = preds[idx_not_nan]
    labels = labels[idx_not_nan]

    return torch.sqrt(criterion(preds, labels))


def compute_loss_on_test(encoder, ode_func, classifier, args, dataloader, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
        dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
        dosing[:, :, 0] = features[:, :, -2]
        dosing = dosing.permute(1, 0, 2)

        encoder_out = encoder(features)
        qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
        z0 = sample_standard_gaussian(qz0_mean, qz0_var)

        solves = z0.unsqueeze(0).clone()
        try:
            for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                z0 += dosing[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0])
                sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        if phase == "test":
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            # print(idx_not_nan)
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            times = times[idx_not_nan.flatten()]
            ptnms += ptnm * len(times)
            Times = torch.cat((Times, times*24))

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))
        
        else:
            """
            time_idx = (times >= 21)
            preds = preds[:, time_idx, :]
            labels = labels[:, time_idx, :]
            """
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            # print(idx_not_nan)
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))

    rmse_loss = mean_squared_error(
        ground_truth.cpu().numpy(), predictions.cpu().numpy(),
        squared=False
    )
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    if phase == "test":
        return {"PTNM": ptnms,
                "TIME": Times,  
                "labels": ground_truth.cpu().tolist(), 
                "preds": predictions.cpu().tolist(),
                "loss": rmse_loss}
    else:
        return {"labels": ground_truth.cpu().tolist(), 
                "preds": predictions.cpu().tolist(),
                "loss": rmse_loss,
                "r2": r2}


def compute_loss_on_interp(encoder, ode_func, classifier, args, dataloader, dataloader_o, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
        ptnm_o, times_o, features_o, labels_o, cmax_time_o = dataloader_o.__next__()
        assert ptnm == ptnm_o

        dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
        dosing[:, :, 0] = features[:, :, -2]
        dosing = dosing.permute(1, 0, 2)

        encoder_out = encoder(features_o)
        qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
        z0 = sample_standard_gaussian(qz0_mean, qz0_var)

        solves = z0.unsqueeze(0).clone()
        try:
            for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                z0 += dosing[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0])
                sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        # print(idx_not_nan)
        preds = preds[idx_not_nan]
        labels = labels[idx_not_nan]

        times = times[idx_not_nan.flatten()]
        ptnms += ptnm * len(times)
        Times = torch.cat((Times, times*24))

        predictions = torch.cat((predictions, preds))
        ground_truth = torch.cat((ground_truth, labels))
        
    rmse_loss = mean_squared_error(
        ground_truth.cpu().numpy(), predictions.cpu().numpy(),
        squared=False
    )
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    return {"PTNM": ptnms,
            "TIME": Times,  
            "labels": ground_truth.cpu().tolist(), 
            "preds": predictions.cpu().tolist(),
            "loss": rmse_loss}

        
        
        
