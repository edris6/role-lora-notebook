#!/bin/bash

# This script is designed to be run inside a Google Colab environment
# after cloning the repository. It sets up the project by installing
# all necessary dependencies.

# Exit immediately if a command exits with a non-zero status to catch errors.
set -e

# Navigate into the repository directory.
# This script assumes the repository was cloned with the name 'role-lora-notebook'.
echo "==> Changing directory into role-lora-notebook..."
cd role-lora-notebook

# Install the required Python packages quietly.
echo "==> Installing dependencies from requirements.txt..."
pip install -q -r requirements.txt

echo "✅ Setup complete!"
echo "You can now open 'role_lora_pipeline.ipynb' from the file browser on the left and run it."