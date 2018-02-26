#!/bin/bash
set -e
set -v

# Install system deps for Python
sudo apt-get update && sudo apt-get install -y make build-essential \
             libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
             libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
             xz-utils tk-dev libturbojpeg && sudo apt-get clean

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Setup env for pyenv
export PYENV_ROOT="${HOME}/.pyenv"
echo "export PATH=\"${PYENV_ROOT}/bin:\$PATH\"" >> ~/.profile
echo "eval \"\$(pyenv init -)\"" >> ~/.profile
echo "eval \"\$(pyenv virtualenv-init -)\"" >> ~/.profile
source ~/.profile

# Install Python and project deps
pyenv install 3.6.3
pyenv virtualenv 3.6.3 raiff-data-cup-3.6.3
pip3 install -r requirements.txt
