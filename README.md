# MedicalMultitaskModeling

MedicalMultitaskModeling (M3) enables training foundational medical imaging models using multi-task learning.

The software is provided on "AS IS" basis, i.e. it comes without any warranty, express or implied including (without limitations) any warranty of merchantability and warranty of fitness for a particular purpose.

Please note that this software is licensed under the LICENSE FOR SCIENTIFIC NON-COMMERCIAL RESEARCH PURPOSES, see [license.md](license.md).

## Installation

```bash
pip install medicalmultitaskmodeling m3-sdk
# Extra dependency groups recommended for development:
pip install "medicalmultitaskmodeling[interactive,testing]" m3-sdk
```

Verify system dependencies and installation:

```python
import cv2; import torch; assert torch.cuda.is_available()
from mmm.interactive import *
from mmm.api.M3Model import M3Model
```

See [pyproject.toml](pyproject.toml) for all available extras.

This project depends on [m3-sdk](https://github.com/FraunhoferMEVIS/m3-sdk).

## Usage and available models

- The `DEFAULT_MODEL` is the model presented in [Overcoming data scarcity in biomedical imaging with a foundational multi-task model](https://doi.org/10.1038/s43588-024-00662-z)
- The weights for `TC_NOPANDA` are connected to [Tissue Concepts: supervised foundation models in computational pathology](https://www.sciencedirect.com/science/article/pii/S0010482524017062?via%3Dihub)
- The `UNICORN_ENCODER` achieved first place in the first foundation model benchmark, UNICORN. Currently, only the 2D vision backbone (encoder) is publicly available.
- The `WSC_MTL_TINY` is a tiny model presented in [Whole Slide Concepts: A Supervised Foundation Model For Pathological Images](https://arxiv.org/abs/2507.05742v3)


For **pure inference** based on a pre-trained model (downloaded automatically), import it as follows:
```python
from mmm.api.M3Model import M3Model, M3_MODELS

from mmm.api.M3Model import DEFAULT_MODEL  # default UMedPT weights
from mmm.api.M3Model import TC_NOPANDA  # default Tissue Concepts weights 
from mmm.api.M3Model import UNICORN_ENCODER  # First Place UNICORN challenge
from mmm.api.M3Model import WSC_MTL_TINY  # default Whole Slide Concepts weights
```

```python
# Load the selected model
model = M3Model(M3_MODELS[UNICORN_ENCODER])
# The individual components can be accessed through keys
print(model.keys())

import torch; import torch.nn as nn
with torch.inference_mode():
    feature_pyramid: list[torch.Tensor] = model["encoder"](torch.rand(1, 3, 224, 224).to(model.device))
    hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])
```

- More examples for deep learning researchers: [M3 examples](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling/tree/main/examples)
- More information for building software with M3: [m3-sdk examples](https://github.com/FraunhoferMEVIS/m3-sdk/tree/main/examples)



## Citation

If you use this project, please cite our work:
- [Overcoming data scarcity in biomedical imaging with a foundational multi-task model](https://doi.org/10.1038/s43588-024-00662-z)
- [Tissue Concepts: supervised foundation models in computational pathology](https://www.sciencedirect.com/science/article/pii/S0010482524017062?via%3Dihub)

```
@article{SchaeferOvercoming2024,
    title = {Overcoming data scarcity in biomedical imaging with a foundational multi-task model},
    journal = {Nature Computational Science},
    issn = {2662-8457},
    doi = {10.1038/s43588-024-00662-z},
    author = {Schäfer, Raphael and Nicke, Till and Höfener, Henning and Lange, Annkristin and Merhof, Dorit and Feuerhake, Friedrich and Schulz, Volkmar and Lotz, Johannes and Kiessling, Fabian},
    year = {2024},
}

@article{nicke2025tissue,
  title={Tissue concepts: Supervised foundation models in computational pathology},
  author={Nicke, Till and Sch{\"a}fer, Jan Raphael and H{\"o}fener, Henning and Feuerhake, Friedrich and Merhof, Dorit and Kie{\ss}ling, Fabian and Lotz, Johannes},
  journal={Computers in biology and medicine},
  volume={186},
  pages={109621},
  year={2025},
  publisher={Elsevier}
}
```
