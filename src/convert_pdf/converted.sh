#!/bin/bash

# --- Function to display usage information ---
usage() {
  echo "Usage: $0 -f <notebook_filename.ipynb>"
  echo "This script converts a Jupyter Notebook (.ipynb) to a PDF."
  echo "It requires the notebook to be in the same directory as the script."
  exit 1
}

# --- Parse command-line arguments ---
while getopts ":f:" opt; do
  case $opt in
    f)
      NOTEBOOK_FILE="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# --- Check if a filename was provided ---
if [ -z "$NOTEBOOK_FILE" ]; then
  echo "Error: Notebook filename not provided."
  usage
fi

# --- Check if the file exists ---
if [ ! -f "$NOTEBOOK_FILE" ]; then
  echo "Error: File '$NOTEBOOK_FILE' not found."
  exit 1
fi

# --- Main conversion logic ---

echo "Converting $NOTEBOOK_FILE to PDF..."

# Get the filename without the extension
FILENAME_NO_EXT="${NOTEBOOK_FILE%.*}"

# Run the conversion using jupyter nbconvert
jupyter nbconvert --to PDF "$NOTEBOOK_FILE"

# Check if the PDF file was successfully created
if [ -f "$FILENAME_NO_EXT.pdf" ]; then
  echo "Conversion successful! PDF is located at: $FILENAME_NO_EXT.pdf"
else
  echo "Conversion failed."
  exit 1
fi