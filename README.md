# MedicalMultitaskModeling

The project enables training foundational medical imaging models using multi-task learning. 

The software is provided on "AS IS" basis, i.e. it comes without any warranty, express or implied including (without limitations) any warranty of merchantability and warranty of fitness for a particular purpose.

Please note that this software is licensed under the LICENSE FOR SCIENTIFIC NON-COMMERCIAL RESEARCH PURPOSES, see license.md.

## Installation:

To install the project and its dependencies, run the following command: 

```bash
pip install medicalmultitaskmodeling
# Including extra dependency groups "interactive" and "testing" recommended for development:
pip install medicalmultitaskmodeling[interactive, testing]

# Verify system dependencies
import cv2; import torch; assert torch.cuda.is_available()
# Verify MMM
from mmm.interactive import *
```

You can check the pyproject.toml file to see all available extras.

## Usage

For **pure inference** based on the pre-trained model (downloads automatically):

```python
# See our tutorial notebooks in the Quick Start Guide for more details.
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier="cuda:0")

import torch; import torch.nn as nn
with torch.inference_mode():
    feature_pyramid: list[torch.Tensor] = model["encoder"](torch.rand(1, 3, 224, 224).to(model.device))
    hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])
```

For a quick start in **model training** or more advanced inference examples, see the "Quickstart Guide" below.

## Quickstart Guide

To begin training multi-task models, you can use our `quickstart.ipynb` getting started notebook.
We recommend using our directory layout as created using our template as following:

1. Install the 'copier' package using pipx:

```bash
# We use pipx to install copier in isolated environment. We use copier to scaffold the code for an experiment. By the time of writing, we used copier version 9.2.0
pipx install copier
```

2. Use the template from a local 'medicalmultitaskmodeling' checkout to create a scaffold for your experiment.

```bash
# To create a new experiment next to your checkout of medicalmultitaskmodeling
copier copy ../medicalmultitaskmodeling/copier_template/ .
```

### Using VSCode development container

3. Open the development container using VSCode via the command `@command:remote-containers.rebuildAndReopenInContainer`. This requires the extension `ms-vscode-remote.remote-containers`.

4. Inside the development container, run the VSCode task (`@command:workbench.action.tasks.runTask`) `Prepare environment` which will reload the window after its done.

5. Run the `quickstart.ipynb` notebook to start your training and learn about this project.

### Using virtualenv

If you prefer to use a virtual environment instead of a container, follow these steps:

1. Create a new virtual environment in your template directory: `virtualenv venv`
1. Activate the virtual environment using `source venv/bin/activate`. For Windows `./venv/Scripts/activate`.
1. Install the 'medicalmultitaskmodeling' package and its dependencies in the virtual environment:

```bash
pip install medicalmultitaskmodeling[interactive]
# Or with a local checkout, and using Jupyterlab:
pip install /your/local/path/medicalmultitaskmodeling[interactive] jupyterlab
```

4. Run the `quickstart.ipynb` notebook. We recommend opening the folder in VSCode. Alternatively, you can use `LOCAL_DEV_ENV=True jupyter lab` and visit the link starting with http://localhost:8888/.

#### System dependencies

We *strongly* recommend using MMM with our public Docker images.
If that is not possible, setup GPU support, check with `nvidia-smi` and run:

```bash
sudo apt install python3-opencv -y
```

## Citation

If you use this project, please cite [our work](https://doi.org/10.1038/s43588-024-00662-z):

```
@article{SchaeferOvercoming2024,
    title = {Overcoming data scarcity in biomedical imaging with a foundational multi-task model},
    journal = {Nature Computational Science},
    issn = {2662-8457},
    doi = {10.1038/s43588-024-00662-z},
    author = {Schäfer, Raphael and Nicke, Till and Höfener, Henning and Lange, Annkristin and Merhof, Dorit and Feuerhake, Friedrich and Schulz, Volkmar and Lotz, Johannes and Kiessling, Fabian},
    year = {2024},
}
```

## Development

1. Start poetry environment `poetry init`
1. Add the package as a git submodule `git submodule add <repository-url>`
1. Add the package, including interactive and dependencies for adding the tests: `poetry add ./medicalmultitaskmodeling/ --editable -E interactive -E testing`
1. For practical examples on how to get started with development, refer to one of our projects, such as [UMedPT](https://github.com/FraunhoferMEVIS/UMedPT).

## Docker images

```bash
# Verify your GPU Docker setup using the hello-world image:
docker run --rm --gpus=all hello-world
# Only system requirements:
MMMVERSION=$(poetry version -s) && docker pull hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:$MMMVERSION
# Verify with
MMMVERSION=$(poetry version -s) && docker run --rm -it --gpus=all hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:$MMMVERSION nvidia-smi
# With dependencies pre-installed:
MMMVERSION=$(poetry version -s) && docker pull hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-stack:$MMMVERSION
```

## Start local infrastructure and inference API with Docker Compose

```bash
# Profiles:
# - inference runs MMM inference container
# - storage runs network drive based on S3 and JuiceFS
# - annotation runs Labelstudio annotation GUI
# - empaia runs infrastructure for gigapixel imaging
MMMVERSION=$(poetry version -s) docker compose --profile inference --profile storage --profile annotation --profile empaia up --build --remove-orphans -d
```

# Repository Structure

For more detailed information, please refer to the docstrings within each directory.

- **torch_ext**: Contains Torch utilities that, while not specific to multi-task learning, can simplify its implementation. This includes our caching utilities.
- **task_sampling**: Provides utilities for enumerating tasks in a way that integrates with PyTorch.
- **inference_api**: starting point to our inference and few-shot-training FastAPI

### data_loading 

This directory contains tools for loading medical data and annotations, supporting formats such as NIfTI, DICOM, and GeoJSON.
It also contains the annotation type specific dataset wrappers such as `SemSegDataset`, responsible for data verification and visualization.

### interactive

This directory has been restructured to allow for easy importing in interactive environments like Jupyter. For instance, you can import several modules with a single line:

```python
from mmm.interactive import blocks, configs as cfs, data, tasks, training, pipes
```

### logging 

Here you'll find utilities that integrate with our logging and visualization tools.

### mtl_modules 

This directory houses multi-task learning types, such as `PyramidEncoder`, and specific tasks.

### neural 

This directory contains PyTorch modules that are not based on our multi-task learning types.

### optimization

This is the home of `MTLOptimizer`. It integrates several PyTorch optimizers with our training strategy and employs the `ZeroRedundancyOptimizer` strategy for distributed training.

### resources

This directory contains static files, like HTML templates for logging.

### trainer

The `Loop` class, used by the `MtlTrainer` class to execute multi-task learning, is located here.
