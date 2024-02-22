Contribution
============

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
