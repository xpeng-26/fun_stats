# Bivariate Gaussian Visualization: Principal Axis vs. Regression Line

This interactive tool visualizes the geometric and statistical differences between the Principal Axis (PCA) and the Regression Line (Conditional Mean) on a 3D bivariate Gaussian surface.

## How to Run on macOS

1.  **Extract the Package**: Unzip the provided folder to your local directory.
2.  **Launch the Application**: Locate the file named **`Run_App.command`** and **double-click** it.
3.  **Access the Dashboard**: Wait for the terminal window to display `Dash is running on http://127.0.0.1:8050`. Open this URL in your web browser (Safari, Chrome, or Firefox).

---

## Troubleshooting: "Permission Denied" or Opening as Text

If double-clicking the `.command` file opens a text editor or shows a permission error, the execution bit was likely lost during transfer. Follow these steps:

1.  Open the **Terminal** app.
2.  Type `chmod +x ` (ensure there is a space after the x).
3.  Drag the **`Run_App.command`** file into the Terminal window to automatically paste its path.
4.  Press **Enter**.
5.  Try double-clicking the file again.

---

## Technical Summary of the Execution Script

When you launch the `.command` script, the following automated steps occur:

* **Directory Initialization**: The script locates the project folder to ensure all relative paths are correct.
* **Virtual Environment Setup**: A local Python virtual environment (`venv`) is created. This ensures that the application dependencies do not interfere with your system-wide Python installation.
* **Dependency Installation**: The script uses `pip` to install the required libraries:
    * **Dash**: The web framework for the interface.
    * **Plotly**: The engine used for high-fidelity 3D rendering.
    * **SciPy/NumPy**: Used for the underlying bivariate normal distribution and eigendecomposition calculations.
* **Server Launch**: The Python script `app.py` is executed, starting a local web server on port 8050.

---

## Interaction Guide

* **Camera Persistence**: You can rotate or zoom the 3D plot using your mouse. The camera angle remains locked even when parameters like Correlation or Standard Deviation are modified.
* **Reset View**: To return to the default top-down perspective, refresh the browser page.
* **Parameter Adjustments**: Use the sliders in the left panel to modify the distribution. The PCA axis and Regression line will update in real-time.

---

## System Requirements

* **Operating System**: macOS 10.15 or later.
* **Python**: Python 3.9 or higher (standard on most modern macOS systems).