# Infinite Dilution Activity Coefficient (IDAC) Prediction Model

This repository contains a PyTorch-based Multi-Layer Perceptron (MLP) model for predicting the infinite dilution activity coefficient (IDAC) of solutes in solvents. The model takes into account basic thermophysical properties of both the solvent and solute, as well as the temperature of the mixture, to make predictions.

## Model Overview

- **Model Architecture**: A simple MLP with two hidden layers.
- **Input Features**: 
  - **Solvent properties**:
    - Critical temperature (Tc)
    - Critical pressure (Pc)
    - Critical volume (Vc)
    - Acentric factor (ω)
    - Dipole moment (μ)
    - Molecular weight (Mw)
  - **Solute properties**:
    - Critical temperature (Tc)
    - Critical pressure (Pc)
    - Critical volume (Vc)
    - Acentric factor (ω)
    - Dipole moment (μ)
    - Molecular weight (Mw)
  - **Mixture property**:
    - Temperature of the mixture (T)

- **Output**: The predicted infinite dilution activity coefficient (IDAC).

- **Reference**: https://doi.org/10.1016/j.fluid.2016.10.033
## Requirements

Before running the model, you need to install the required dependencies. You can do this using `pip` and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch > 2.0
- NumPy
- Pandas
- Scikit-learn (for data preprocessing and evaluation metrics)

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/idac-prediction.git
    ```
2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the model**:
    You can use the `main.py` script to make predictions. First, ensure that you have the input data formatted correctly. \
   Correct format for the input vector: \
    `[Tc_solute, Pc_solute, Vc_solute, omega_solute, dipole_solute, MW_solute, Tc_solvent, Pc_solvent, Vc_solvent, omega_solvent, dipole_solvent, MW_solvent].`
