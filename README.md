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
from mmm.api.M3Model import M3Model, MMM_MODELS, DEFAULT_MODEL
model = M3Model(MMM_MODELS[DEFAULT_MODEL], device_identifier="cuda:0")

import torch; import torch.nn as nn
with torch.inference_mode():
    feature_pyramid: list[torch.Tensor] = model["encoder"](torch.rand(1, 3, 224, 224).to(model.device))
    hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])
```

## Using the UNICORN 1st place solution

This project achieved first place in the first foundation model benchmark, UNICORN! To download and use the model:

```python
from mmm.api.M3Model import M3Model, MMM_MODELS, UNICORN_ENCODER
model = M3Model(MMM_MODELS[UNICORN_ENCODER], device_identifier="cuda:0")
```

## Citation

If you use this project, please cite our work:
- [Overcoming data scarcity in biomedical imaging with a foundational multi-task model](https://doi.org/10.1038/s43588-024-00662-z)
- [Tissue Concepts: supervised foundation models in computational pathology](https://arxiv.org/abs/2409.03519)

```
@article{SchaeferOvercoming2024,
    title = {Overcoming data scarcity in biomedical imaging with a foundational multi-task model},
    journal = {Nature Computational Science},
    issn = {2662-8457},
    doi = {10.1038/s43588-024-00662-z},
    author = {Schäfer, Raphael and Nicke, Till and Höfener, Henning and Lange, Annkristin and Merhof, Dorit and Feuerhake, Friedrich and Schulz, Volkmar and Lotz, Johannes and Kiessling, Fabian},
    year = {2024},
}

@article{nicke2024tissue,
  title={Tissue Concepts: supervised foundation models in computational pathology},
  author={Nicke, Till and Schaefer, Jan Raphael and Hoefener, Henning and Feuerhake, Friedrich and Merhof, Dorit and Kiessling, Fabian and Lotz, Johannes},
  journal={arXiv preprint arXiv:2409.03519},
  doi = {10.48550/arXiv.2409.03519}
  year={2024}
}
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
