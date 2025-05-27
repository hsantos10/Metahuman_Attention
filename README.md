# Metahuman_Attention

## Overview
This project focuses on analyzing human motion data using deep learning techniques. It processes marker data (TRC files) and associated kinematic or dynamic data (MOT/STO files) to train an LSTM (Long Short-Term Memory) model with an Attention mechanism. The primary goal is to predict forces, kinematics, or other biomechanical variables based on motion capture inputs. The system also supports the integration of subject-specific metadata like weight and height to potentially improve prediction accuracy.

## Features
- **Data Ingestion:** Loads and processes common biomechanics file formats:
    - TRC (Track Row Column) files for 3D marker trajectories.
    - MOT and STO files for kinematics (e.g., joint angles) and dynamics (e.g., joint moments, ground reaction forces).
- **Data Preprocessing:**
    - Normalization (MinMax, Standard Scaler) of data.
    - Padding and truncation of sequences to handle variable lengths.
    - Option to remove specified markers.
    - Imputation of missing data points.
- **Exploratory Data Analysis (EDA):**
    - Calculation and visualization of correlation matrices for TRC and other data types.
    - Cross-correlation analysis between different data streams (e.g., TRC vs. MOT).
- **Deep Learning Model:**
    - LSTM network architecture tailored for sequence data.
    - Integrated Attention mechanism (Additive or Dot-Product) to focus on relevant parts of the input sequences.
    - Bidirectional LSTM option for capturing context from both past and future time steps.
- **Model Training & Evaluation:**
    - Training and validation pipelines with detailed progress reporting (Loss, MAE, R²).
    - Masking of padded sequences during loss calculation and metric evaluation.
    - Saving and loading of trained models (standard PyTorch `.pt` and ExecuTorch `.pte` formats).
- **Results Analysis:**
    - Validation of the model on a test set.
    - Calculation of MAE, MSE, RMSE, and R² scores per feature.
    - Plotting of training and validation metrics over epochs.
- **Metadata Integration:**
    - Ability to load subject-specific metadata (e.g., weight, height) from a CSV file.
    - Integration of metadata into the model's input features.

## File Structure
- **`main.py`**: The main script to run the entire workflow, from data loading to model training and evaluation. Contains key configuration parameters.
- **`data_loading.py`**: Contains functions for reading, processing, and preparing TRC, MOT, STO, and metadata files. Handles data splitting, padding, and normalization.
- **`model.py`**: Defines the LSTM network architectures, including the simple LSTM and the LSTM with Attention (Additive and Dot-Product).
- **`train.py`**: Includes functions for training the model, evaluating it on a validation set, performing final validation, and analyzing results.
- **`eda.py`**: Provides functions for exploratory data analysis, such as generating correlation matrices and cross-correlation plots.
- **`utils.py`**: Contains utility functions, including marker name mappings, normalization functions, and helper functions for file reading (e.g., improved STO file reader).
- **`LICENSE`**: Contains the project's license information.
- **`README.md`**: This file, providing an overview and documentation for the project.
- **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
- **`.gitattributes`**: Defines attributes per path.

## Model Details
The core of this project is an LSTM network designed for sequence-to-sequence prediction tasks on motion data.
- **LSTM Layers:** The number of LSTM layers can be configured.
- **Hidden Size:** The dimensionality of the LSTM hidden states can be set.
- **Bidirectional:** The LSTM can be configured to be bidirectional, allowing information flow from both past and future time steps.
- **Attention Mechanism:**
    - **Purpose:** To allow the model to weigh the importance of different time steps in the input sequence when making predictions for each time step in the output sequence.
    - **Types:**
        - `dot`: Dot-product attention.
        - `additive`: Additive attention (Bahdanau-style).
    - **Masking:** The attention mechanism correctly handles padded sequences by masking padding tokens during the computation of attention scores.
- **Output Layer:** A fully connected layer maps the (potentially attention-weighted) LSTM outputs to the desired prediction dimension.

## Contributing
Contributions to this project are welcome. Please ensure that any code contributions are well-documented and tested.

## License
Refer to the `LICENSE` file for details on the licensing of this project.

## Setup and Usage

### Dependencies
This project requires Python and the following libraries:
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install these dependencies using pip. It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install torch pandas numpy scikit-learn matplotlib seaborn
```
(Note: Ensure you install a PyTorch version compatible with your system, especially if you have a CUDA-enabled GPU. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for specific installation commands.)

### Configuration
1.  **File Paths:** Before running the project, you need to configure the data file paths in `main.py`. Update the following variables with the correct paths to your data:
    *   `TRC_FILE_PATH`: Path pattern for TRC files (e.g., `r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\MarkerData\*.trc'`).
    *   `DATA_FILE_PATH`: Path pattern for MOT or STO files (e.g., `r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\ForcesData\*.mot'`).
    *   `METADATA_FILE_PATH` (if `USE_METADATA_LOADING` is `True`): Path to the CSV file containing subject metadata (e.g., `r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\Subject_info_DAVID.csv'`).

2.  **Data Columns:** In `main.py`, you can specify which columns from your data files (`.mot`/`.sto`) to use as target variables for the model:
    *   `data_columns_to_use`: Set this to a list of integer column indices (e.g., `[0, 1, 2]`) or set it to `None` to use all data columns.

3.  **Metadata Usage:**
    *   `USE_METADATA_LOADING`: Set this boolean variable in `main.py` to `True` if you want to load and integrate subject metadata (weight, height) into the model. If `True`, ensure `METADATA_FILE_PATH` and related parameters (prefixes, suffixes for ID extraction) are correctly set.

### Running the Project
Execute the `main.py` script to start the data loading, preprocessing, EDA, model training, and evaluation pipeline:
```bash
python main.py
```
The script will print progress information to the console, and generated plots (e.g., correlation matrices, training curves) will be displayed or saved to files. Trained models (`best_trained_model.pt` and `best_trained_model.pte`) will also be saved in the project's root directory.
