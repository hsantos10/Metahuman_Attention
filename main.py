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
# # +++ START EXECUTORCH IMPORTS +++
# from torch.export import export, ExportedProgram
# from executorch.exir import EdgeProgramManager, to_edge
# from executorch.sdk.etdump import GeneratableEtDump, ETDumpGen # For ETDump, if needed
# from executorch.sdk.etrecord import ETRecord # For ETRecord, if needed
# # +++ END EXECUTORCH IMPORTS +++

from data_loading import (
    read_trc_file, process_trc_files, process_data_files, read_data_file,
    prepare_data_for_modeling, prepare_data_for_modeling_with_metadata # import the metadata  if used
)
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
#METADATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\Subject_info_DAVID.csv' # Update with  actual path
TRC_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\MarkerData\*.trc'
#DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\IK_Results\*.mot'  # Kinematics
#DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\ID_Results\*.sto'  # Dynamics
DATA_FILE_PATH = r'G:\Shared drives\Digital Twin Model\Pilot Test dataset\ToDavid\ForcesData\*.mot'  # Ground Forces

#Select data columns (adjust as needed)
data_columns_to_use = None  # If you want to use all data columns, set data_columns_to_use = None
#data_columns_to_use = ['pelvis_tx','pelvis_ty','pelvis_tz']  # Example
#data_columns_to_use = [2]  # Example
#data_columns_to_use =list(range(6))

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
        'Left Elbow Lateral Epicondyle',
        'Left Wrist Radial Styloid',
        'Right Elbow Lateral Epicondyle',
        'Right Wrist Radial Styloid',
        'Left Anterior Superior Iliac Spine',
        'Left Posterior Superior Iliac Spine',
        'Right Posterior Superior Iliac Spine',
        'Right Anterior Superior Iliac Spine',
        'Mid-Chest marker',
        'Sensor Left',
        'Sensor Right',
        'Left Ankle Medial Malleolus',
        'Right Ankle Lateral Malleolus'
   ]  # Define markers to remove
    
    # 2. Prepare Data for Modeling --- CHOOSE WHICH DATA PREPARATION TO USE --- 
    USE_METADATA_LOADING = False # Set to False to use the older prepare_data_for_modeling

    if USE_METADATA_LOADING:
        print("Using data preparation with metadata.")
        metadata_df = load_metadata(METADATA_FILE_PATH, subject_id_col='Subject', weight_col='Weight', height_col='Height')
        if metadata_df is None:
            print("Failed to load metadata. Exiting.")
            exit()
       
        # Define prefixes and suffixes for subject ID extraction -  ADAPT THESE CAREFULLY
        # Example: if TRC files are "Experiment Detail.xlsx - subj00.trc"
        # and metadata 'Subject' column has "subj00" or "00"
        # And data files are "subj00_forces.mot"
        trc_file_prefix = "Experiment Detail.xlsx - subj" # Adjust if only "subj" or if it varies
        trc_file_suffix = ".trc"
        data_file_prefix = "subject_trial_forces_" # Or "subj" or "" depending on data_files names
        _, first_data_file_ext = os.path.splitext(data_files[0])
        data_file_suffix = first_data_file_ext # e.g. ".mot" or ".sto"
       
        #data_columns_to_use = list(range(6)) # Example: use first 6 columns from data files as targets
       
        (train_data_tuple, val_data_tuple, test_data_tuple,
         max_length, input_size, output_size) = prepare_data_for_modeling_with_metadata(
            trc_files, data_files, metadata_df, data_columns_to_use,
            trc_file_prefix=trc_file_prefix, trc_file_suffix=trc_file_suffix,
            data_file_prefix=data_file_prefix, data_file_suffix=data_file_suffix,
            subject_id_col_in_metadata='Subject', # Ensure this matches your CSV
            weight_col_in_metadata='Weight',   # Ensure this matches your CSV
            height_col_in_metadata='Height', # Ensure this matches your CSV
            test_size=0.1, val_size=0.1, random_state=42
        )
        # train_data_tuple is (train_trc_seqs, train_data_seqs, train_input_masks)
       
    else:
        print("Using data preparation WITHOUT metadata (original flow).")
        processed_trc_data = process_trc_files(trc_files, markers_to_remove)
        if not processed_trc_data:
            print("Error processing TRC files. Exiting.")
            exit()
        processed_data = process_data_files(data_files) # These are targets
        if not processed_data:
            print("Error processing data files (targets). Exiting.")
            exit()
        
        #data_columns_to_use = list(range(6)) # Example: use first 6 columns from data files as targets
       
        (train_data_tuple, val_data_tuple, test_data_tuple,
         max_length, input_size, output_size) = prepare_data_for_modeling(
            processed_trc_data, processed_data, data_columns_to_use,
            test_size=0.1, val_size=0.1, random_state=42
        )
        # train_data_tuple is (train_trc_seqs, train_data_seqs, train_input_masks)



    processed_trc_data = process_trc_files(trc_files, markers_to_remove)


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


    
    # Unpack data tuples
    train_trc_seqs, train_data_seqs, train_input_masks = train_data_tuple
    val_trc_seqs, val_data_seqs, val_input_masks = val_data_tuple
    test_trc_seqs, test_data_seqs, test_input_masks = test_data_tuple
    
    # Check if data loading was successful
    if not train_trc_seqs or not val_trc_seqs or not test_trc_seqs:
        print("Error: One or more data splits are empty after processing. Exiting.")
        exit()
    if input_size == 0 or output_size == 0:
        print(f"Error: input_size ({input_size}) or output_size ({output_size}) is zero. Check data loading and column selection. Exiting.")
        exit()
    
    # Prepare data for modeling
    (train_data, val_data, test_data, max_length, input_size, output_size) = prepare_data_for_modeling(
        processed_trc_data, processed_data, data_columns_to_use)
    
    # Create TensorDatasets and DataLoaders
    # train_dataset = TensorDataset(torch.stack(train_data[0]), torch.stack(train_data[1]))
    # val_dataset = TensorDataset(torch.stack(val_data[0]), torch.stack(val_data[1]))
    # test_dataset = TensorDataset(torch.stack(test_data[0]), torch.stack(test_data[1]))
    
    # Create TensorDatasets and DataLoaders
    # The TensorDataset will now include masks
    train_dataset = TensorDataset(torch.stack(train_trc_seqs), torch.stack(train_data_seqs), torch.stack(train_input_masks))
    val_dataset = TensorDataset(torch.stack(val_trc_seqs), torch.stack(val_data_seqs), torch.stack(val_input_masks))
    test_dataset = TensorDataset(torch.stack(test_trc_seqs), torch.stack(test_data_seqs), torch.stack(test_input_masks))
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # 4. Model Training and Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Network_with_Attention(input_size, hidden_size=512, output_size=output_size,attention_type='dot').to(device)
    #model = LSTM_Network(input_size, hidden_size=512, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 150
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

        # # --- Convert and Save the ExecuTorch Model ---
        # print(f"\n--- Converting model to ExecuTorch .pte format ---")
        
        # # 1. Get an example input.
        # #    ExecuTorch's export process needs an example input to trace the model.
        # #    You can get this from your DataLoader or create a dummy tensor
        # #    with the correct shape and type.
        # #    Let's take one batch from the test_loader.
        # example_inputs, _ = next(iter(test_loader))
        # example_input_for_export = example_inputs[0:1].to(device) # Use a single sample from the batch

        # # Make sure the model is on CPU for export if it was on GPU
        # # Or ensure your capture_config handles device correctly
        # model_cpu = model.to("cpu")
        # example_input_for_export = example_input_for_export.to("cpu")
        
        # # 2. Export to ATen/Edge dialect using torch.export
        # #    You might need to adapt this based on your model's specifics (e.g., dynamic shapes)
        # print("Exporting the model using torch.export...")
        # aten_exported_program: ExportedProgram = export(model_cpu, (example_input_for_export,))
        # print("Model exported to ATen dialect.")

        # # 3. Convert to Edge IR
        # print("Converting to Edge IR...")
        # edge_program_manager: EdgeProgramManager = to_edge(aten_exported_program)
        # print("Model converted to Edge IR.")
        
        # # 4. Convert to ExecuTorch IR and save as .pte
        # #    This step performs the final conversion to the format that can run on device.
        # print("Converting to ExecuTorch IR and serializing to .pte file...")
        # executorch_program = edge_program_manager.to_executorch()
        
        # with open(executorch_model_save_path, "wb") as f:
        #     f.write(executorch_program.buffer)
        # print(f"ExecuTorch model saved successfully to {executorch_model_save_path}")

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