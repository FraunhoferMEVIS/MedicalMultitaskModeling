Contribution
============

Debugging and development
-------------------------

- Debugging in Jupyter only works without PyTorch Multiprocessing (num_workers=0 for all cohorts!)
- In Streamlit, rerunning works well even when the Python debugger is attached.

Formatting
----------

Use Black, for example by adding this to .vscode/settings.json::


    "black-formatter.args": [
        "--line-length",
        "120"
    ],
    "[python]": {
        ...
        "editor.defaultFormatter": "ms-python.black-formatter"
    }

Guidelines
----------

- Use doctests if the test is readable and instructive. If not, put the test in the `tests` folder
- Instead of directly using `requires_grad = False` to disable optimization of a shared block,
consider using `block.freeze_all_parameters(True)`. It handles more things like setting eval mode.

Preparing datasets
------------------

PyTorch has a known issue when reading from numpy arrays with type objects or the built-in types such as dict or list
in the `__getitem__` method. Convert them to numpy arrays or Tensor in `__init__` as done in the imagenet example.
This is only needed for large datasets.

Troubeshooting
==============

When using instance norm:

ValueError: Expected more than 1 spatial element when training, got input size torch.Size([INFEATURES, OUTFEATURES, 1, 1])

Your architecture downsamples your inputs too much and instance norm cannot normalize given only one value.