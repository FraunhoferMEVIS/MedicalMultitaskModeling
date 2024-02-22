Fraunhofer MEVIS Internal
=========================

Cluster infrastructure
----------------------

For quick download times of images, in the internal cluster infrastructure there is a mirror: `https://registry.fme.lan/mtl-torch`.


Working with confidential data
------------------------------

We integrated W&B and the imaging tasks upload image data for logging purposes.
For confidential data you can turn off auto-sync to the global WANDB instance using `WANDB_MODE=offline`.
Then, W&B will create the local folder with your data without automatically syncing.
Afterwards, start a local W&B instance (this will be a single `docker` command) to view your logs.

Working with internal Gitlab packages
-------------------------------------

For using private packages you need to set the environment variables:

- POETRY_HTTP_BASIC_fhggitlab_USERNAME
- POETRY_HTTP_BASIC_fhggitlab_PASSWORD

Usually, devcontainer files are good examples for working configurations.

Gitlab CI
---------

First, create new runner: https://gitlab.cc-asp.fraunhofer.de/mevis-histo/tissue-concepts/medicalmultitaskmodeling/-/runners/new

```bash
docker run --rm -v /srv/gitlab-runner/config:/etc/gitlab-runner gitlab/gitlab-runner register \
  --non-interactive \
  --url "https://gitlab.cc-asp.fraunhofer.de" \
  --token "$GITLAB_RUNNER_AUTH_TOKEN" \
  --executor "docker" \
  --docker-image alpine:latest \
  --description "docker-runner" \
  --docker-privileged
  --docker-volumes "/certs/client"
# For GPU, add to /srv/gitlab-runner/config/config.toml
# [runners.docker]
#     gpus = "all" or "device=2,4" for selecting the third and fifth GPU
docker restart gitlab-runner
```

```bash
docker run -d --name gitlab-runner --restart always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /srv/gitlab-runner/config:/etc/gitlab-runner \
    gitlab/gitlab-runner:latest
```