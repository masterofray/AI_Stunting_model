To use a bash script with an argument like `-f` for the input filename, you can create a shell script that uses a `while` loop to parse command-line options. This allows you to specify the notebook to convert when you run the script.

### Create the Shell Script

You can save the following code in a file, for example, `convert_nb.sh`, and make it executable. The script will handle the input argument and run the conversion command. Take a look `converted.sh` file.


### How to Use This Script in Colab

1.  **Mount Google Drive:** First, you must manually run the Python cell in your Colab notebook to mount your Google Drive. This step cannot be automated by the bash script.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Create the Script and Install Dependencies:** In a new Colab code cell, run the following commands to create the `convert_nb.sh` file and install the necessary LaTeX dependencies.

3.  **Run the Script:** After the `convert_nb.sh` file is created and the dependencies are installed, you can use a separate cell to copy your notebook from Google Drive and then run the script to convert it to a PDF.

The script will then find the notebook in the current directory, convert it to a PDF, and leave the PDF file there for you to download from the Colab file explorer.