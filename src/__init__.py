"""PIGNet2: protein-ligand interaction prediction model.

Importable as the ``pignet2`` package (``src/`` is mapped to ``pignet2`` in
``pyproject.toml``). Core subpackages:

    from pignet2.models import PIGNet, PIGNetMorse
    from pignet2.data import ComplexDataModule, complex_to_data

The scripts under ``exe/`` are intended to be run as standalone scripts (they
rely on flat, sys.path-based imports) and are not part of the import surface.
"""

__version__ = "2.0.0"
