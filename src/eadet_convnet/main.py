import argparse
from datetime import datetime
import pathlib
import random
import sys

import numpy as np
import xarray as xr

import torch
from torch.backends import cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary

from config import read_config
from data import get_train_test_partition, get_train_test_targets, CustomDataset
from jobs import train, evaluate, test
from model import MultiTaskConvNet


g_output_fstem = "MultiTaskConvNet"


def main():
    parser = argparse.ArgumentParser(
        description='Micro-seismic earthquake detection: "Training" Step'
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        dest="config_file",
        help="configuration file to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="W",
        help="how many subprocesses to use for data loading (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        dest="use_cuda",
        action="store_false",
        default=True,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.8,
        metavar="R",
        help="the portion of the dataset used for training (default: 0.8)",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        default=False,
        help="shuffle the dataset before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="seed Python and Pytorch RNGs (default: None)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        default=False,
        help="test final model and save predictions and ground-truth arrays",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="save the current (trained) model",
    )
    parser.add_argument(
        "--save-costs",
        action="store_true",
        default=False,
        help="write train and test cost values to an ascii column file",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="write summarized info on current model to a plain text file",
    )
    parser.add_argument(
        "--logfile",
        action="store_true",
        default=False,
        help="write logfile for current run to a plain text file",
    )

    args = parser.parse_args()

    # ----------
    tic = datetime.now()
    tic_str = tic.strftime("%Y-%m-%d_%H-%M")

    # CUDA for PyTorch
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Set seed for RNGs
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)  # for current GPU
        torch.cuda.manual_seed_all(args.seed)  # for all GPUs

    # Improve performance by enabling benchmarking feature
    # see: https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True

    # ----------
    # Read dataset config file
    config = read_config(args.config_file)
    images_dirname = pathlib.Path(config.images_dirname)
    images_paths_pattern = str(images_dirname.joinpath(config.images_paths_pattern))
    images_filename_tmpl = str(images_dirname.joinpath(config.images_filename_tmpl))

    targets_filename = config.norm_params_filename

    # ----------
    # kwargs for DataLoader
    dataloader_kwargs = {"num_workers": args.num_workers, "shuffle": True}

    if use_cuda:
        dataloader_kwargs["pin_memory"] = True

    # Dataset partitioning + prepare targets
    partition = get_train_test_partition(
        images_paths_pattern, args.train_split_ratio, args.random_split
    )

    train_targets, test_targets = get_train_test_targets(targets_filename, partition)

    # Training-set generator
    train_loader = DataLoader(
        dataset=CustomDataset(images_filename_tmpl, partition["train"], train_targets),
        batch_size=args.batch_size,
        **dataloader_kwargs,
    )

    # Test-set generator
    test_loader = DataLoader(
        dataset=CustomDataset(images_filename_tmpl, partition["test"], test_targets),
        batch_size=args.test_batch_size,
        **dataloader_kwargs,
    )

    # ----------
    # Create the model object
    model = MultiTaskConvNet().to(device)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    class_criterion = nn.BCEWithLogitsLoss()
    bbox_criterion = nn.MSELoss()

    train_costs, test_costs = [], []
    for epoch in range(1, args.epochs + 1):
        # Training step
        train_cost = train(
            model,
            optimizer,
            class_criterion,
            bbox_criterion,
            train_loader,
            device,
            epoch,
            args,
        )

        # Evaluation step
        test_cost = evaluate(
            model, class_criterion, bbox_criterion, test_loader, device
        )

        print("\nTrain-set epoch cost : {:.9f}".format(train_cost))
        print("Test-set average loss: {:.9f}\n".format(test_cost))

        train_costs.append(train_cost)
        test_costs.append(test_cost)

    # ----------
    fstem_dated = "{}_{}".format(g_output_fstem, tic_str)

    # Save trained model
    if args.save_model:
        filename_model = fstem_dated + "_model.pth"
        torch.save(model.state_dict(), filename_model)

    # Save cost values
    if args.save_costs:
        filename_costs = fstem_dated + "_cost.nc"
        costs = xr.DataArray(
            np.vstack((train_costs, test_costs)).T,
            dims=["epoch_index", "epoch_cost"],
            coords={"epoch_cost": ["train", "test"]},
        )
        costs.to_netcdf(filename_costs)

    # Save model predictions
    if args.test_model:
        predictions, groundtruth = test(model, test_loader, device, args)
        predictions.to_netcdf(fstem_dated + "_pred.nc")
        groundtruth.to_netcdf(fstem_dated + "_true.nc")

    # ----------
    toc = datetime.now()

    # Write log-file
    if args.logfile:
        filename_logfile = fstem_dated + "_log.txt"
        with open(filename_logfile, "w") as fid:
            fid.write("\n".join(sys.argv[1:]))
            fid.write("\n")

            if args.save_model:
                fid.write("Trained model: {}\n".format(filename_model))

            fid.write("Started at: {}\n".format(tic))
            fid.write("Finished at: {}\n".format(toc))

    # Write model summary
    if args.summary:
        dummy_model = MultiTaskConvNet().to(device)
        filename_summary = fstem_dated + "_summary.txt"
        original_stdout = sys.stdout

        with open(filename_summary, "w") as fid:
            fid.write(str(dummy_model))
            fid.write("\n" * 3)

            # NOTE: `device` arg should be `str` not `torch.device`
            sys.stdout = fid
            images_batch, labels_batch = next(iter(test_loader))
            input_shape = images_batch.shape[1:]
            summary(
                dummy_model, input_shape, batch_size=args.batch_size, device=device.type
            )

            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
