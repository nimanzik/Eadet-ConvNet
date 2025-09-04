import glob
import os.path as op
import random

import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset


g_image_dtype = torch.float32


def get_train_test_partition(data_paths_pattern, train_split_ratio, random_split):
    """
    Parameters
    ----------
    data_paths_pattern : str
        Data pathnames pattern containing shell-style wild cards. This
        pattern is given to `glob.glob` built-in method to find all
        path names that match it.
    train_split_ratio : float
        The portion of the dataset used for training. Must be between 0.0
        and 1.0 (example: 0.8 means 80% of data is used for training).
    random_split : bool
        If True, it shuffles the dataset before splitting. If False, then
        the dataset is split in an ordered fashion.

    Returns
    -------
    partition : dict
        Creates a dictionary called `partition` where we gather:
          - in `partition['train']` a list of training IDs,
          - in `partition['test']` a list of test IDs.

    References
    ----------
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    # Image filename template: `<example_id>.nc`. Example: EV000001.nc

    ID_list = []
    for fn in glob.iglob(data_paths_pattern):
        example_id = op.splitext(op.basename(fn))[0]
        ID_list.append(example_id)

    ID_list.sort()
    if random_split:
        random.shuffle(ID_list)

    n_data = len(ID_list)
    n_train = int(train_split_ratio * n_data)  # Train/dev splitting

    partition = {"train": ID_list[:n_train], "test": ID_list[n_train:]}

    return partition


def get_train_test_targets(targets_filename, partition):
    """
    CAUTION: this is a helper function defined for this specific case.

    Creates a dictionary called `targets` where for each ID of the dataset,
    the associated target is given by `targets[ID]`.

    Parameters
    ----------
    targets_filename : str
        Filename (netCDF format) in which targets (ground-truth values)
        are stored. In our case, targets are seismic-source parameters.
    partition : dict
        See function `~get_train_test_partition` output.

    Returns
    -------
    a : tuple of 2 dicts
        Elements of `a` refer to: a[0] -> train targets, a[1] -> test targets.
        Each element of `a` is a dictionary, whose keys are sample IDs and
        values are corresponding `numpy.ndarray` of target values. Example:
        {'SEC00002': array([ 0.55,  0.14,  1.68, ... , 0.73]),
         'SEC13009': array([ 0.86,  0.16,  0.72, ... , 0.47]), ...}
    """

    # Columns are (p, bx, by, bw, bh)
    params_da = xr.load_dataarray(targets_filename)

    a = []
    for step in ("train", "test"):
        step_IDs = partition[step]

        # `xarray` supports vectorized indexing
        step_da = params_da.sel(id=step_IDs)
        step_targets = dict(zip(step_da.get_index("id"), step_da.values))

        a.append(step_targets)

    return tuple(a)


class CustomDataset(Dataset):
    """Characterizes a dataset for PyTorch."""

    def __init__(self, filename_template, ID_list, targets):
        """
        Parameters
        ----------
        filename_template : str
            Path template used to fetch and load examples (images).
        ID_list : list
            List of example IDs (see the output of the function
            `get_train_test_partition`).
        targets : dict
            A dictionary that maps example IDs to corresponding
            `numpy.ndarray` of target values (see the output of the
            function `~get_train_test_targets`).
        """
        self.filename_template = filename_template
        self.targets = targets
        self.ID_list = ID_list

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.ID_list)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        # Select sample ID
        ID = self.ID_list[idx]

        # Load image, reshape to (1, nreceivers, ntimes), and normalise
        X = np.load(self.filename_template % dict(example_id=ID))
        X = torch.tensor(X[None, :, :], dtype=g_image_dtype)
        X /= X.abs().max().item()

        y = self.targets[ID]
        y = torch.tensor(y, dtype=g_image_dtype)

        return (X, y)
