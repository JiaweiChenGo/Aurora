r"""
Modified from:
  GLUE: https://github.com/gao-lab/GLUE;
  TOSICA: https://github.com/JackieHanLab/TOSICA
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version


name = "Aurora"
__version__ = version(name)

from .model import load_model

from .fit_model import fit_model

from .dataset import configure_dataset



