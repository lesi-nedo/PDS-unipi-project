#!/bin/bash

REMOTE_USER="o.nedobiychuk"
REMOTE_IP="spmcluster.unipi.it"
REMOTE_PORT="22"
REMOTE_DIR="/home/o.nedobiychuk/project"
CURRENT_DIR="$(cd "$(dirname "$BASH_SOURCE[0]")" && pwd)"

cd ../ || exit 1

echo "Current directory: $(pwd)"

# Path to the SSH key file
SCRIPT_DIR="$PWD"
KEY_FILE="${SCRIPT_DIR}/key/pds_key"
SSH_OPTIONS="-p $REMOTE_PORT -i $KEY_FILE -o ServerAliveInterval=32000"

echo "Current directory after moving up: $(pwd)"
echo "DIR: $CURRENT_DIR"

ITEMS_TO_TRANSFER=(
  "$CURRENT_DIR/pyproject.toml"
  "$CURRENT_DIR/run_dt.sh"
  "$CURRENT_DIR/README.md"
  "$CURRENT_DIR/project2.pdf"
  "$CURRENT_DIR/CMakeLists.txt"
  "$CURRENT_DIR/src"
  "$CURRENT_DIR/results"
  "$CURRENT_DIR/report"
  "$CURRENT_DIR/logs"
  "$CURRENT_DIR/include"
  "$CURRENT_DIR/external"
  "$CURRENT_DIR/data"
  "$CURRENT_DIR/analysis"
)

RSYNC_OPTIONS=(
  "-avz"
  "--progress"
  "--exclude=__pycache__"
  "--exclude=.git"
  "--exclude=.idea"
  "--exclude=*.map"
)

set -e

item_exists() {
  if [ ! -e "$1" ]; then
    echo "Error: $1 does not exist"
    return 1
  fi
  return 0
}

check_key_file() {
  if [ ! -f "$KEY_FILE" ]; then
    echo "Error: SSH key file does not exist at $KEY_FILE"
    exit 1
  fi
  if [ "$(stat -c %a "$KEY_FILE")" != "600" ]; then
    echo "Warning: Incorrect permissions for SSH key file. It should be 600. Changing permissions..."
    chmod 600 "$KEY_FILE"
  fi
}

setup_ssh_agent() {
  # Check if ssh-agent is already running
  if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "Starting ssh-agent..."
    eval "$(ssh-agent -s)"
  fi
  
  # Check if key is already added
  if ! ssh-add -l | grep -q "$KEY_FILE"; then
    echo "Adding SSH key to agent (you'll need to enter passphrase once)..."
    ssh-add "$KEY_FILE"
  else
    echo "SSH key already loaded in agent"
  fi
}

echo "Checking if the remote server is reachable..."
ssh $SSH_OPTIONS "$REMOTE_USER@$REMOTE_IP" "echo 'Remote server is reachable'"
if [ $? -ne 0 ]; then
  echo "Error: Unable to reach the remote server. Please check your connection."
  exit 1
fi


echo "Starting file transfer to remote machine using rsync..."

# Call the check_key_file function
check_key_file

# Setup SSH agent to avoid repeated passphrase prompts
setup_ssh_agent

for item in "${ITEMS_TO_TRANSFER[@]}"; do
  if item_exists "$item"; then
    echo "Transferring $item..."
    rsync "${RSYNC_OPTIONS[@]}" -e "ssh $SSH_OPTIONS" "$item" "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR"
    if [ $? -eq 0 ]; then
      echo "Successfully transferred $item"
    else
      echo "Error: failed to transfer $item"
    fi
  fi
done

set +e
echo "File transfer complete"
