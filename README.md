# MedicalMultitaskModeling

The project enables training foundational medical imaging models using multi-task learning.

## Installation:

To install the project and its dependencies, run the following command: 

```bash
pip install medicalmultitaskmodeling
# Including extra dependency groups "interactive" and "testing" recommended for development:
pip install medicalmultitaskmodeling[interactive, testing]
```

You can check the pyproject.toml file to see all available extras.

### System dependencies

We *strongly* recommend using MMM with our public Docker images.
If that is not possible, setup GPU support, check with `nvidia-smi` and run:

```bash
sudo apt install python3-opencv -y
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
docker pull hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:1.0.0
# Verify with
docker run --rm -it --gpus=all hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:1.0.0 nvidia-smi
# With dependencies pre-installed:
docker pull hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-stack:1.0.0
```

## Citation

If you use this project, please cite [our work](https://doi.org/10.48550/arXiv.2311.09847):

```
@misc{schäfer2023overcoming,
      title={Overcoming Data Scarcity in Biomedical Imaging with a Foundational Multi-Task Model}, 
      author={Raphael Schäfer and Till Nicke and Henning Höfener and Annkristin Lange and Dorit Merhof and Friedrich Feuerhake and Volkmar Schulz and Johannes Lotz and Fabian Kiessling},
      year={2023},
      eprint={2311.09847},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
