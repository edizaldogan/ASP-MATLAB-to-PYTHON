# ASP-MATLAB-to-PYTHON
Python adaptations of the MATLAB companion file to the book *Applied Signal Processing* by T. Dutoit and F. Marques, Springer 2008.

## Project Structure
* `audio_samples/`: Directory containing source `.wav` files (e.g., `speech.wav`). 
Ensure this remains one level above the working scripts.

## Installation
1. **Clone the repository and install Python libraries:**
```bash
git clone https://github.com/edizaldogan/ASP-MATLAB-to-PYTHON.git
```

```bash
cd ASP-MATLAB-to-PYTHON
```

```bash
pip install -r requirements.txt
```

2. **Install the Jupyter VS Code Extension:**
To view the output figures and execute the code cell-by-cell, install the official **Jupyter** extension within Visual Studio Code.

## Usage
These scripts are formatted with `# %%` markers and are designed to be run interactively on a cell-by-cell basis. 
This allows you to view the output figures side-by-side with the code.

1. Open the project folder in Visual Studio Code.
2. Navigate to the relevant subfolder (e.g., `chapters/ch-1/translation-clean-python`) and open the target script.
3. Execute the code cell-by-cell by clicking the **Run Cell** text above each block, or by pressing the `Shift + Enter` keyboard shortcut.

> **Note:** Because later cells depend on variables calculated earlier, executing cells out of order will result in errors. Always run the cells sequentially, or utilize the **Run Above** button for quicker execution.
