#!/bin/bash

START_SSH_MODE=${START_SSH_MODE:-"blocking"}
SSHD_PORT=${SSHD_PORT:-24}
echo $SSH_PUBLIC_KEY >> ~/.ssh/authorized_keys

printenv > ~/.ssh/environment
if [ "$START_SSH_MODE" = "blocking" ]; then
  /usr/sbin/sshd -D -p $SSHD_PORT "$@"
elif [ "$START_SSH_MODE" = "service" ]; then
   echo "Port $SSHD_PORT" >> /etc/ssh/sshd_config
   service ssh start
else
    echo "Unknown START_SSH_MODE: $START_SSH_MODE"
    exit 1
fi
