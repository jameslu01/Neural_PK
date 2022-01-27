import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from torchdiffeq import odeint_adjoint as odeint


def merge_predictions(evals_per_fold, reference_data):
    cols = ["PTNM", "TIME", "preds", "labels"]
    left = evals_per_fold[0][cols]
    for right in evals_per_fold[1:]:
        left = left.merge(right[cols], on=["PTNM", "TIME", "labels"], how="left")
    preds = [col for col in left.columns.values if col.startswith("preds")]
    left["pred_agg"] = left[preds].agg("mean", axis=1)

    ref = reference_data[["PTNM", "DSFQ"]].drop_duplicates()
    left = left.merge(ref, on="PTNM", how="left")
    # get rid of the first round of treatment
    left_q1w = left[(left.DSFQ == 1) & (left.TIME >= 168)]
    left_q3w = left[(left.DSFQ == 3) & (left.TIME >= 504)]
    return pd.concat([left_q1w, left_q3w], ignore_index=False)


def sample_standard_gaussian(mu, sigma):
    device = torch.device("cpu")
    if mu.is_cuda:
        device = mu.get_device()

    d = torch.distributions.normal.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def compute_loss_on_train(criterion, labels, preds):
    preds = preds.permute(1, 0, 2)
    idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
    preds = preds[idx_not_nan]
    labels = labels[idx_not_nan]
    return torch.sqrt(criterion(preds, labels))


def compute_loss_on_test(encoder, ode_func, classifier, tol, latent_dim, dataloader, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)

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
                sol = odeint(ode_func, z0, time_interval, rtol=tol, atol=tol)
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnm)
            continue

        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        if phase == "test":
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            times = times[idx_not_nan.flatten()]
            ptnms += ptnm * len(times)
            Times = torch.cat((Times, times * 24))

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))

        else:
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))

    rmse_loss = mean_squared_error(ground_truth.cpu().numpy(), predictions.cpu().numpy(), squared=False)
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    if phase == "test":
        return {
            "PTNM": ptnms,
            "TIME": Times,
            "labels": ground_truth.cpu().tolist(),
            "preds": predictions.cpu().tolist(),
            "loss": rmse_loss,
            "r2": r2,
        }
    else:
        return {"labels": ground_truth.cpu().tolist(), "preds": predictions.cpu().tolist(), "loss": rmse_loss, "r2": r2}
