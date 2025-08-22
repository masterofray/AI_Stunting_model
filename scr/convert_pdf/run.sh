#!/bin/bash

# Execute the script
chmod +x converted.sh

# Install dependencies for PDF conversion
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-latex-extra pandoc

# Copy the notebook from Google Drive to the local Colab directory
cp "/content/drive/MyDrive/Colab Notebooks/Mamberamo_District_XGBoost.ipynb" ./

# Run the script with the -f argument
bash ./converted.sh -f "Mamberamo_District_XGBoost.ipynb"