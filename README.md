# ASP-MATLAB-to-PYTHON
Python adaptations of the MATLAB companion file to the book *Applied Signal Processing* by T. Dutoit and F. Marques, Springer 2008.

## Project Structure
* `audio_samples/`: Directory containing source `.wav` files (e.g., `speech.wav`). 
Ensure this remains one level above the working scripts.

## Installation & Usage
**Clone the repository and install Python libraries:**

Open your terminal in the directory where you want the project folder to be created, then run:

```bash
git clone https://github.com/edizaldogan/ASP-MATLAB-to-PYTHON.git
```

Navigate into the newly created repository folder:

```bash
cd ASP-MATLAB-to-PYTHON
```

Run the following command to install the necessary libraries and kernel.

```bash
pip install -r requirements.txt
```

The main scripts are formatted with `# %%` markers and are designed to be run interactively on a cell-by-cell basis. 
This allows you to view the output figures side-by-side with the code.

1. Open the project folder in Visual Studio Code.

    > **Note:** Jupyter VS Code Extension must be installed.
    To view the output figures and execute the code cell-by-cell, install the official **Jupyter** extension within Visual Studio Code.
    A recommendation on the bottom right will appear if this extension is not installed; simply click on the prompt to install it.

2. Navigate to the relevant subfolder (e.g., `chapters/ch-X/translation-clean-python`) and open the target script.
3. Execute the code cell-by-cell by clicking the **Run Cell** text above each block, or by pressing the `Shift + Enter` keyboard shortcut.

> **Note:** Because later cells depend on variables calculated earlier, executing cells out of order will result in errors. Always run the cells sequentially, or utilize the **Run Above** button for quicker execution.

## Streamlit Tools
In addition to the cell-by-cell scripts, this repository includes interactive Streamlit apps.

To launch the apps:

1. Open your terminal and ensure you are in the directory of the project (`ASP-MATLAB-to-PYTHON`).
2. Run the application by passing the script's path to the `streamlit run` command. For example:
   ```bash
   streamlit run chapters/ch-X/streamlit_app.py
   ```
3. The dashboard will automatically open in your default web browser, where you can adjust parameters, and view the visualizations in real-time.