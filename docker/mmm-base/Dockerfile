FROM ubuntu:22.04

# It is not recommended to use the root user for Python stuff
# Here you can configure another user for development
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and clean up afterwards
RUN apt-get update -qq \
  && apt-get install --no-install-recommends -y wget gpg ca-certificates software-properties-common \
  && wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com jammy main" | tee /etc/apt/sources.list.d/hashicorp.list \
  && apt-get update -qq \
  && apt-get install --no-install-recommends -y \
  gcc \
  curl \
  python3-dev \
  python3-pip \
  python3-opencv \
  openssh-server \
  sudo \
  git \
  nomad \
  pipx \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && ln -s /usr/bin/python3 /usr/bin/python

# Set default shell to bash
ENV SHELL=/bin/bash
SHELL [ "/bin/bash", "-c" ]


# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# SSH preparations (root user required)
RUN mkdir -p /run/sshd && mkdir -p /var/run/sshd \
  && echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config \
  && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER devuser
# ENV PATH="${PATH}:/home/devuser/.local/bin/"
# # If you need the root user, uncomment the following lines
RUN mkdir -p ~/.ssh
ENV PIP_ROOT_USER_ACTION=ignore
ENV PATH="${PATH}:/root/.local/bin/"

RUN pipx ensurepath
RUN pipx install "uv"

RUN echo 'export PATH="${PATH}:/root/.local/bin/"' >> /root/.bashrc
# Set the default editor to vscode
RUN echo 'export EDITOR="code --wait"' >> /root/.bashrc
# Add git completion
RUN echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

# Scripts
COPY sshd-dev.sh /scripts/sshd-dev.sh
RUN chmod +x /scripts/sshd-dev.sh
