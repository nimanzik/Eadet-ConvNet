import numpy as np
import xarray as xr

import torch


def train(
    model, optimizer, class_criterion, bbox_criterion, train_loader, device, epoch, args
):
    """
    Train model for a given epoch.

    Returns
    -------
    cost : float
        Running cost (i.e. training cost for the current epoch).
    """

    # Size of training set (m)
    trainset_size = len(train_loader.dataset)

    # Number of mini-batches
    n_batches = len(train_loader)

    # Switch to train mode
    model.train()

    # Loop over training mini-batches
    train_cost = 0.0
    for i_batch, (data, target) in enumerate(train_loader, 0):
        # Transfer to GPU
        data, target = data.to(device), target.to(device)
        c_target = target[:, [0]]
        b_target = target[:, [1, 2, 3, 4]]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        c_out, b_out = model(data)
        c_loss = class_criterion(c_out, c_target)
        b_loss = bbox_criterion(b_out, b_target)
        loss = c_loss + b_loss
        loss.backward()
        optimizer.step()

        # Size of current mini-batch
        batch_size = data.shape[0]

        # Sum up the current mini-batch loss
        train_cost += loss.item()

        # Log the status
        status_num = i_batch + 1
        if (status_num % args.log_interval == 0) or (status_num == n_batches):
            print(
                f"Train Epoch: {epoch:3d} "
                f"[{status_num * batch_size:6d}/{trainset_size:6d} "
                f"({status_num / n_batches * 100:3.0f}%)] "
                f"Batch Loss: {loss.item():.9f}"
            )

    train_cost /= trainset_size
    return train_cost


def evaluate(model, class_criterion, bbox_criterion, test_loader, device):
    """
    Evaluate/validate model on dev/test set.

    Returns
    -------
    cost : float
        Validation cost for the current state of the CNN model.
    """

    # Switch to evaluate mode
    model.eval()

    test_cost = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            c_target = target[:, [0]]
            b_target = target[:, [1, 2, 3, 4]]

            # Prediction
            c_pred, b_pred = model(data)

            # Sum up batch loss
            c_loss = class_criterion(c_pred, c_target)
            b_loss = bbox_criterion(b_pred, b_target)
            loss = c_loss + b_loss
            test_cost += loss.item()

    testset_size = len(test_loader.dataset)
    test_cost /= testset_size

    return test_cost


def test(model, test_loader, device, args, apply_mcd=False):
    """
    Test model on test dataset.

    Parameters
    ----------
    apply_mcd : bool, default: False
        Whether to apply Monte-Carlo dropout at station level.
    """

    # Switch to evaluate mode
    model.eval()

    # Apply Monte Carlo dropout if required
    if apply_mcd:
        model.s_dropout.force_dropout = True

    # Extract model predictions + ground-truth
    output_size = 5
    testset_size = len(test_loader.dataset)
    predictions = np.zeros((testset_size, output_size), dtype=np.float32)
    groundtruth = np.zeros_like(predictions)

    # Needed for `xr.DataArray`
    da_dims = ["id", "labels"]
    da_coords = {"labels": "c bx by  bw bh".split()}

    with torch.no_grad():
        for i_batch, (data, target) in enumerate(test_loader):
            data = data.to(device)
            c_target = target[:, [0]]
            b_target = target[:, [1, 2, 3, 4]]

            c_pred, b_pred = model(data)

            i1 = i_batch * args.test_batch_size
            i2 = i1 + args.test_batch_size

            predictions[i1:i2, [0]] = c_pred.cpu().numpy()
            predictions[i1:i2, [1, 2, 3, 4]] = b_pred.cpu().numpy()

            groundtruth[i1:i2, [0]] = c_target.numpy()
            groundtruth[i1:i2, [1, 2, 3, 4]] = b_target.numpy()

    predictions = xr.DataArray(predictions, dims=da_dims, coords=da_coords)
    groundtruth = xr.DataArray(groundtruth, dims=da_dims, coords=da_coords)

    return (predictions, groundtruth)


__all__ = """
    train
    evaluate
    test
""".split()
