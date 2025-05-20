# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:30:20 2025

@author: David

Main Script Organization: The main.py script provides a clear workflow:
Load and preprocess data.
Perform EDA.
Prepare data for modeling.
Train and evaluate the model.
Validate and analyze the results.

"""

# main.py
import glob
import torch
import torch.nn as nn
import torch.optim as optim

# FROM PYTORCH LITE: from torch.utils.mobile_optimizer import optimize_for_mobile
# +++ START EXECUTORCH IMPORTS +++
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge
from executorch.sdk.etdump import GeneratableEtDump, ETDumpGen # For ETDump, if needed
from executorch.sdk.etrecord import ETRecord # For ETRecord, if needed
# +++ END EXECUTORCH IMPORTS +++

from data_loading import read_trc_file, process_trc_files, process_data_files, read_data_file, prepare_data_for_modeling  # Import the new function
from eda import calculate_correlation_matrix, calculate_cross_correlation
from model import LSTM_Network, LSTM_Network_with_Attention
from train import train_and_evaluate, validate_model, analyze_results, plot_results
from utils import normalize_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# Create DataLoaders
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os


# --- File Paths ---

TRC_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\MarkerData\*.trc'
#DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\IK_Results\*.mot'  # Kinematics
#DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\ID_Results\*.sto'  # Dynamics
DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\ForcesData\*.mot'  # Ground Forces

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load and Preprocess Data
    trc_files = sorted(glob.glob(TRC_FILE_PATH))
    data_files = sorted(glob.glob(DATA_FILE_PATH))
    print(f"Number of TRC files found: {len(trc_files)}")
    print(f"Number of data files found: {len(data_files)}")

    # Process TRC files to ensure consistency AND remove markers
    markers_to_remove = [
       'Box1', 'Box2', 'Box3', 'Box4', 'DL1', 'DL2', 'DL3', 'DH1', 'DH2', 'DH3', 'T1', 'T2',
       #'LELBL', 'LWRL', 'RELBL', 'RWRL', 'LASI', 'LPSI', 'RPSI', 'RASI', 'MChest', 'SENL', 'SENR', 'LANKM', 'RANKL'
   ]  # Define markers to remove
    processed_trc_data = process_trc_files(trc_files, markers_to_remove)
    if not processed_trc_data:
        print("Error processing TRC files. Exiting.")
        exit()

    # Process .mot or .sto files to ensure consistency
    processed_data = process_data_files(data_files)
    if not processed_data:
        print("Error processing data files. Exiting.")
        exit()

    # Load a single TRC and data file for EDA (using the first file for simplicity)
    trc_data_for_eda = processed_trc_data[0].copy()  # Use processed data!
    data_for_eda = read_data_file(data_files[0])
    
    # %%

    # # 2. Exploratory Data Analysis (EDA)
    # if trc_data_for_eda is not None and data_for_eda is not None:
    #     print("Performing EDA...")

    #     # Normalize data before correlation analysis
    #     trc_data_for_eda_normalized = normalize_data(trc_data_for_eda.copy(), type='standard')
    #     data_for_eda_normalized = normalize_data(data_for_eda.copy(), type='standard')

    #     calculate_correlation_matrix(trc_data_for_eda_normalized, file_type="TRC")
    #     calculate_correlation_matrix(data_for_eda_normalized, file_type="DATA")  # Generic type
    #     calculate_cross_correlation(trc_data_for_eda_normalized, data_for_eda_normalized)
    # else:
    #     print("Could not perform EDA due to data loading issues.")
        
    # %%

    # 3. Prepare Data for Modeling -  Select data columns (adjust as needed)
    data_columns_to_use = None  # If you want to use all data columns, set data_columns_to_use = None
    #data_columns_to_use = ['pelvis_tx','pelvis_ty','pelvis_tz']  # Example
    data_columns_to_use = [1]  # Example
    data_columns_to_use =list(range(5))
    
    # Prepare data for modeling
    (train_data, val_data, test_data, max_length, input_size, output_size) = prepare_data_for_modeling(
        processed_trc_data, processed_data, data_columns_to_use)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.stack(train_data[0]), torch.stack(train_data[1]))
    val_dataset = TensorDataset(torch.stack(val_data[0]), torch.stack(val_data[1]))
    test_dataset = TensorDataset(torch.stack(test_data[0]), torch.stack(test_data[1]))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # 4. Model Training and Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Network_with_Attention(input_size, hidden_size=512, output_size=output_size,attention_type='dot').to(device)
    #model = LSTM_Network(input_size, hidden_size=512, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 200
    train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    plot_results(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s)
    
    print("Training complete.")

    # --- Saving the Standard PyTorch Model and PyTorch Lite Model ---
    model_save_path = "best_trained_model.pt"  # Unified name for the standard model
    # CHANGED: .pte extension for ExecuTorch
    executorch_model_save_path = "best_trained_model.pte" 
    
    try:
        model.eval() # Ensure model is in evaluation mode

        # Save the Standard PyTorch Model (optional, but good for reference)
        torch.save(model.state_dict(), model_save_path) # Save state_dict for standard model
        print(f"Standard model state_dict saved successfully to {model_save_path}")

        # --- Convert and Save the ExecuTorch Model ---
        print(f"\n--- Converting model to ExecuTorch .pte format ---")
        
        # 1. Get an example input.
        #    ExecuTorch's export process needs an example input to trace the model.
        #    You can get this from your DataLoader or create a dummy tensor
        #    with the correct shape and type.
        #    Let's take one batch from the test_loader.
        example_inputs, _ = next(iter(test_loader))
        example_input_for_export = example_inputs[0:1].to(device) # Use a single sample from the batch

        # Make sure the model is on CPU for export if it was on GPU
        # Or ensure your capture_config handles device correctly
        model_cpu = model.to("cpu")
        example_input_for_export = example_input_for_export.to("cpu")
        
        # 2. Export to ATen/Edge dialect using torch.export
        #    You might need to adapt this based on your model's specifics (e.g., dynamic shapes)
        print("Exporting the model using torch.export...")
        aten_exported_program: ExportedProgram = export(model_cpu, (example_input_for_export,))
        print("Model exported to ATen dialect.")

        # 3. Convert to Edge IR
        print("Converting to Edge IR...")
        edge_program_manager: EdgeProgramManager = to_edge(aten_exported_program)
        print("Model converted to Edge IR.")
        
        # 4. Convert to ExecuTorch IR and save as .pte
        #    This step performs the final conversion to the format that can run on device.
        print("Converting to ExecuTorch IR and serializing to .pte file...")
        executorch_program = edge_program_manager.to_executorch()
        
        with open(executorch_model_save_path, "wb") as f:
            f.write(executorch_program.buffer)
        print(f"ExecuTorch model saved successfully to {executorch_model_save_path}")

    except Exception as e:
        print(f"Error during model saving or ExecuTorch conversion: {e}")
        import traceback
        traceback.print_exc()
     
    # 5. Model Validation and Analysis
    predictions, actuals = validate_model(model, test_loader, device)
    print(f"actuals:\n {actuals}")
    print(f"predictions:\n {predictions}")
     
    # Load a MOT file to get the correct column names (replace with your actual loading)
    example_mot_file = data_files[0]  # Or however you access your MOT file
    if data_columns_to_use == None:
        column_names = read_data_file(example_mot_file).columns
    else:
        column_names = read_data_file(example_mot_file).columns[data_columns_to_use].tolist()

        
    results_df = analyze_results(predictions, actuals, column_names)
    
    print("\nMain script execution complete.")