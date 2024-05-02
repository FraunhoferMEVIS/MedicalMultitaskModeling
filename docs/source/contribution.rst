Contribution
============

Git Troubleshooting
-------------------

- I merged an external feature branch into my local feature branch. Now the external feature branch was squashed and merged into main. How do I update my local feature branch?
  - Start a new branch from main and apply patches like `git diff main..your_local_feature_branch > patch.diff` and `git apply patch.diff`.

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
