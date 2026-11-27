"""scView backend."""

from scview.core import h5_names as _h5_names

# Names holding "/" are stored escaped in every h5ad (core/h5_names.py).
_h5_names.install_on_import()
