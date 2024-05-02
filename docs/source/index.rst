Getting started
===============

A good place to start is the `quickstart.ipynb`.

Preparing datasets
------------------

PyTorch has a known issue when reading from numpy arrays with type objects or the built-in types such as dict or list
in the `__getitem__` method. Convert them to numpy arrays or Tensor in `__init__` as done in the imagenet example.
This is only needed for large datasets.

Debugging and development
-------------------------

- Debugging in Jupyter only works without PyTorch Multiprocessing (num_workers=0 for all cohorts!)
- In Streamlit, rerunning works well even when the Python debugger is attached.


.. toctree::
  :maxdepth: 3

  contribution.rst

Code documentation
==================

.. toctree::
  :maxdepth: 3

  auto/modules.rst


Troubeshooting
==============

When using instance norm:

ValueError: Expected more than 1 spatial element when training, got input size torch.Size([INFEATURES, OUTFEATURES, 1, 1])

Your architecture downsamples your inputs too much and instance norm cannot normalize given only one value.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
