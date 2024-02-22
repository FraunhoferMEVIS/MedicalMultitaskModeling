Getting started
===============

A good place to start is the `mini_example.ipynb`.
In `mini_example.ipynb`, a shared encoder and a shared decoder are trained using multi-task learning (MTL).
The shared neural modules are called blocks.
Generalizing shared blocks should be the result of the "pre-training" stage
where multiple tasks are optimized simultaneously to find a robust and general set of weights.
After computing such a set of weights using the training set,
we want to assess the generalizability of the shared blocks:

* after each training loop (a.k.a. epoch), a validation is performed for each task **independently**.
* the user can provide hooks which are evaluated after each epoch.
  For example, you might want to evaluate the performance of a random forest applied to a neural encoder's output.
* from time to time, a **downstream evaluation** is performed.

Preparing datasets
------------------

PyTorch has a known issue when reading from numpy arrays with type objects or the built-in types such as dict or list
in the `__getitem__` method. Convert them to numpy arrays or Tensor in `__init__` as done in the imagenet example.
This is only needed for large datasets.

Downstream evaluation
---------------------

Sometimes during training it is important to assess the generalizability of the shared blocks on new tasks.
For this purpose, the user can define *downstream tasks*.
Downstream tasks are expected to be fully differentiable.
As a result, they can be used to fine-tune the shared blocks.

The trainer expects an `MTLTask` object for running downstream tasks.
In consequence, it consists of a training and a validation set.
Similarly to pre-training tasks, a validation set is sampled randomly if none is provided.
If the user needs to evaluate the weights of a downstream task on a test set,
the trainer allows to specify custom code using the `@trainer.callback_foreach_downstream_task()` decorator.


.. toctree::
  :maxdepth: 3

  contribution.rst
  fme.rst

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
