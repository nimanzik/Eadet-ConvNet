import pathlib
from os import path as op

from pyrocko.guts import Object, String


guts_prefix = "pf"


class HasPaths(Object):
    path_prefix = String.T()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._basepath = None

    def set_basepath(self, basepath):
        self._basepath = basepath

    def get_basepath(self):
        assert self._basepath is not None
        return self._basepath

    def get_relpath(self):
        basepath = self.get_basepath()
        return op.realpath(op.join(basepath, self.path_prefix))


class Path(String):
    pass


class TrainingDatasetConfig(HasPaths):
    images_dirname = Path.T(help="Inputs -> Directory name where images are stored")

    norm_params_filename = Path.T(
        help="Targets -> File with source parameters normalized"
    )

    images_paths_pattern = String.T(
        default="*.nc",
        help="Images pathnames pattern containing shell-style wild cards",
    )

    images_filename_tmpl = String.T(
        default="%(event_name)s.nc",
        help='Images filename template containing "%(event_name)s" placeholder',
    )


def read_config(path_to_config):
    # Set full path
    path_to_config = pathlib.Path(op.realpath(path_to_config))
    assert path_to_config.is_file() is True

    # Load and set basepath
    config = TrainingDatasetConfig.load(filename=path_to_config.as_posix())
    config.validate(regularize=True)
    config.set_basepath(path_to_config.parent.as_posix())

    # Relative path (raises error if basepath is not set)
    relpath = pathlib.Path(config.get_relpath())

    # Set path attributes to full path
    for attr_name in ("images_dirname", "norm_params_filename"):
        # Create full path as PosixPath
        p = relpath.joinpath(getattr(config, attr_name))
        setattr(config, attr_name, p.as_posix())

    return config
