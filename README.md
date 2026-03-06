# MedicalMultitaskModeling

MedicalMultitaskModeling (M3) enables training foundational medical imaging models using multi-task learning.

The software is provided on "AS IS" basis, i.e. it comes without any warranty, express or implied including (without limitations) any warranty of merchantability and warranty of fitness for a particular purpose.

Please note that this software is licensed under the LICENSE FOR SCIENTIFIC NON-COMMERCIAL RESEARCH PURPOSES, see [license.md](license.md).

## Installation

```bash
pip install medicalmultitaskmodeling
# Extra dependency groups recommended for development:
pip install "medicalmultitaskmodeling[interactive,testing]"
```

Verify system dependencies and installation:

```python
import cv2; import torch; assert torch.cuda.is_available()
from mmm.interactive import *
```

See [pyproject.toml](pyproject.toml) for all available extras.

## Usage

For **pure inference** based on the pre-trained model (downloads automatically):

```python
from mmm.api.M3Model import M3Model, M3_MODELS, DEFAULT_MODEL
model = M3Model(M3_MODELS[DEFAULT_MODEL])

import torch; import torch.nn as nn
with torch.inference_mode():
    feature_pyramid: list[torch.Tensor] = model["encoder"](torch.rand(1, 3, 224, 224).to(model.device))
    hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])
```

- More examples for deep learning researchers: [M3 examples](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling/tree/main/examples)
- More information for building software with M3: [m3-sdk examples](https://github.com/FraunhoferMEVIS/m3-sdk/tree/main/examples)

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
  doi = {10.48550/arXiv.2409.03519},
  year={2024}
}
```
