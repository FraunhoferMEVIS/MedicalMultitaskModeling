FROM ubuntu:22.04

# It is not recommended to use the root user for Python stuff
# Here you can configure another user for development
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies and clean up afterwards
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y wget gpg ca-certificates \
 && wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com jammy main" | tee /etc/apt/sources.list.d/hashicorp.list \
 && apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    gcc \
    python3-dev \
    python3-pip \
    python3-opencv \
    openssh-server \
    sudo \
    git \
    nomad \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/python3 /usr/bin/python \
 && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -

# Set default shell to bash
ENV SHELL=/bin/bash
SHELL [ "/bin/bash", "-c" ]

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
 && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER devuser

# SSH for root user preparations
RUN mkdir -p /var/run/sshd \
 && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
 && echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
EXPOSE 22

# Install base requirements and poetry
ENV PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt /requirements.txt
RUN python -m pip --no-cache-dir install --upgrade pip \
  && python -m pip --no-cache-dir install -r /requirements.txt \
  && rm /requirements.txt
RUN apt update -qq && apt install curl -y && curl -sSL https://install.python-poetry.org | python3 -

# Put poetry which is in /root/.local/bin/ on path for the root user
ENV PATH="${PATH}:/root/.local/bin/"
# Same change, but for SSH sessions
RUN echo 'export PATH="${PATH}:/root/.local/bin/"' >> /root/.bashrc
# Set the default editor to vscode
RUN echo 'export EDITOR="code --wait"' >> /root/.bashrc
# Add git completion
RUN echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

RUN poetry config virtualenvs.create false

# S3 tools
RUN curl -sSL https://github.com/kahing/goofys/releases/latest/download/goofys -o /root/.local/bin/goofys && chmod +x /root/.local/bin/goofys
RUN curl -sSL https://d.juicefs.com/install | sh -
