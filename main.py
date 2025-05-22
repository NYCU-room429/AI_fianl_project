# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import librosa
# import os
# from multiprocessing import Pool, cpu_count
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import utils


# def process_track_data(row_dict):
#     track_id = row_dict["track"]
#     mix_path = row_dict["mix"]
#     metadata_path = row_dict["metadata"]
#     stem_file_paths = row_dict["stems"]
#     midi_file_paths = row_dict["midis"]

#     track_dict = {}
#     track_dict[track_id] = {
#         "mix": utils.read_flac(mix_path),
#         "metadata": utils.read_metadata(metadata_path),
#         "stems": {
#             os.path.splitext(os.path.basename(stem_flac))[0]: utils.read_flac(stem_flac)
#             for stem_flac in stem_file_paths
#         },
#         "midis": {
#             os.path.splitext(os.path.basename(midi_mid))[0]: utils.read_midi(midi_mid)
#             for midi_mid in midi_file_paths
#         },
#     }
#     return track_dict


# if __name__ == "__main__":
#     instruments_class = utils.read_instruments_class(
#         "slakh-utils\\midi_inst_values\\general_midi_inst_0based.json"
#     )

#     print("================")
#     print(len(instruments_class))
#     print("================")
#     train_dir = "slakh2100_flac_redux\\slakh2100_flac_redux\\train"
#     train_data = utils.load_dataset(train_dir)

#     row_dicts = train_data.to_dict(orient="records")

#     # with Pool(processes=cpu_count()) as pool:
#     #     results = list(
#     #         tqdm(
#     #             pool.imap(process_track_data, row_dicts),
#     #             total=len(row_dicts),
#     #         )
#     #     )

#     # track_dicts = [r for r in results if r is not None]

#     # final_dict = {}
#     # for d in track_dicts:
#     #     final_dict.update(d)

#     print(process_track_data(row_dicts[0]))

#     track_dict = process_track_data(row_dicts[0])
#     track_id = list(track_dict.keys())[0]
#     mel_spectrogram = track_dict[track_id]["mix"]

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         mel_spectrogram, sr=22050, hop_length=512, x_axis="time", y_axis="mel"
#     )
#     plt.colorbar(format="%+2.0f dB")
#     plt.title("Mel-frequency Spectrogram")
#     plt.tight_layout()
#     plt.show()
# main.py

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm # For progress bars
import torch.nn as nn
import torch.optim as optim
# Use specific imports from utils and CRNN
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score # Multi-label metrics
# Remove unused imports: StandardScaler, mean_squared_error, mean_absolute_error, r2_score, multiprocessing, Pool, cpu_count, scipy.stats, matplotlib.pyplot, librosa

import os # Keep os for path handling and cpu_count

import utils # Import your utils.py script
from CRNN import MusicInstrumentDataset, CRNN, INSTRUMENT_MAPPING_PATH, MAX_MEL_FRAMES # Import classes and necessary constants from CRNN.py


# --- Configuration Parameters ---
# Paths
# <--- *** Change this path to your dataset root directory ***
DATASET_ROOT_PATH = 'slakh2100_flac_redux/slakh2100_flac_redux/train'
# INSTRUMENT_MAPPING_PATH is imported from CRNN.py

MODEL_SAVE_PATH = 'instrument_crnn_model.pth'

# Data parameters (Constants like MAX_MEL_FRAMES are imported from CRNN.py)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SPLIT_RATIO = 0.2 # Ratio for validation/test split
RANDOM_STATE = 42 # For reproducible splits


# --- Main Execution ---

if __name__ == "__main__":
    # The initial check for instruments_class length can remain for info
    instruments_mapping_check = utils.read_instruments_class(INSTRUMENT_MAPPING_PATH)
    print("================")
    print(f"Number of instruments in mapping file: {len(instruments_mapping_check)}")
    print("================")

    # --- 1. Load Dataset Information ---
    # utils.load_dataset only scans the directories and gets file paths
    print(f"Loading dataset information from {DATASET_ROOT_PATH}...")
    dataset_df = utils.load_dataset(DATASET_ROOT_PATH)
    print(f"Found {len(dataset_df)} tracks.")

    if len(dataset_df) == 0:
        print("No tracks found in the dataset directory. Please check DATASET_ROOT_PATH.")
        exit()

    # --- 2. Create Dataset Instance and Load Data ---
    # The MusicInstrumentDataset class handles reading audio/midi and preprocessing into memory
    print(f"\nCreating and loading data into MusicInstrumentDataset (MAX_MEL_FRAMES={MAX_MEL_FRAMES})...")
    try:
        # Pass the dataframe and mapping path to the Dataset
        full_dataset = MusicInstrumentDataset(
            dataframe=dataset_df,
            instrument_mapping_path=INSTRUMENT_MAPPING_PATH,
            max_mel_frames=MAX_MEL_FRAMES # Pass the constant
        )
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        exit()
    except FileNotFoundError as e:
         print(f"Error: Required file not found during dataset creation - {e}")
         exit()
    except Exception as e:
        print(f"An unexpected error occurred during dataset creation: {e}")
        exit()


    if len(full_dataset) == 0:
        print("No valid data samples prepared by the dataset. Check file availability, paths, and processing errors in utils and dataset loading.")
        exit()

    # --- 3. Determine Model Input/Output Shapes from Dataset ---
    # Get shapes from the first sample after data loading
    # Dataset item shape is [1, N_MELS, MAX_MEL_FRAMES] for input, [num_classes] for label
    sample_input, sample_label = full_dataset[0]
    input_channel = sample_input.shape[0] # Should be 1
    n_mels = sample_input.shape[1]
    max_mel_frames_actual = sample_input.shape[2] # This should match MAX_MEL_FRAMES
    num_classes = len(sample_label) # Number of output classes = number of instrument classes

    print(f"\nModel Input Shape (C, H, W): ({input_channel}, {n_mels}, {max_mel_frames_actual})")
    print(f"Number of output classes: {num_classes}")
    # Optionally print instrument classes defined by the dataset
    # print(f"Target instrument classes: {full_dataset.instrument_classes}")


    # --- 4. Split Dataset ---
    # Get indices for splitting
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        # Note: Stratifying multi-label data is complex. Skipping for simplicity.
        # If needed, look into techniques like Iterative Stratification.
        # stratify=full_dataset.labels
    )

    # Create subset datasets using indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- 5. Create DataLoaders ---
    # Use os.cpu_count() for potentially faster loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() or 1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() or 1, pin_memory=True)
    print(f"Using {os.cpu_count() or 1} worker processes for DataLoaders. Pin_memory enabled.")


    # --- 6. Build the CRNN Model ---
    print("\nBuilding CRNN model...")
    # Pass the determined shapes to the model constructor
    model = CRNN(
        input_channel=input_channel,
        n_mels=n_mels,
        max_mel_frames=max_mel_frames_actual,
        num_classes=num_classes
    )
    print("Model built.")

    # --- 7. Define Loss Function and Optimizer ---
    # Binary Cross-Entropy Loss for multi-label classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 8. Set up Device (CPU/GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device for training: {device}")

    # --- 9. Training Loop ---
    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for inputs, labels in train_loader_tqdm:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item()) # Display current batch loss

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        all_labels = [] # Store true labels
        all_predictions = [] # Store predicted probabilities

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]", leave=False)

        with torch.no_grad(): # No gradient calculation during validation
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs) # Outputs are probabilities after sigmoid

                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                all_labels.append(labels.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy()) # Append probabilities


        val_epoch_loss = val_running_loss / len(val_dataset)

        # Calculate multi-label metrics from collected labels and predictions
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Apply a threshold (e.g., 0.5) to get binary predictions for metrics
        THRESHOLD = 0.5
        binary_predictions = (all_predictions > THRESHOLD).astype(float)

        # Calculate metrics (using 'micro' average as an example)
        val_precision_micro = precision_score(all_labels, binary_predictions, average='micro', zero_division=0)
        val_recall_micro = recall_score(all_labels, binary_predictions, average='micro', zero_division=0)
        val_f1_micro = f1_score(all_labels, binary_predictions, average='micro', zero_division=0)
        # Calculate binary accuracy (element-wise match rate)
        val_binary_accuracy = np.mean(all_labels == binary_predictions)


        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, "
              f"Val Binary Acc: {val_binary_accuracy:.4f}, "
              f"Val Micro Prec: {val_precision_micro:.4f}, "
              f"Val Micro Rec: {val_recall_micro:.4f}, "
              f"Val Micro F1: {val_f1_micro:.4f}")

    print("\nTraining finished.")

    # --- 11. Save the Model ---
    # Save the model's state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model state dictionary saved to {MODEL_SAVE_PATH}")

    # --- Example of Loading and Prediction (Optional - Can add this back if needed) ---
    # print("\nLoading saved model for testing...")
    # # Need to re-instantiate the model with the correct architecture parameters
    # loaded_model = CRNN(input_channel=input_channel, n_mels=n_mels, max_mel_frames=max_mel_frames_actual, num_classes=num_classes)
    # loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # loaded_model.to(device) # Move to the same device as used during training/inference
    # loaded_model.eval() # Set to evaluation mode
    # print("Model loaded.")

    # # Example prediction on the first sample of the validation set
    # print(f"\nPredicting on a sample from validation set (Index: {val_indices[0]})...")
    # sample_input_tensor, true_label_tensor = full_dataset[val_indices[0]]
    # sample_input_tensor = sample_input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # with torch.no_grad():
    #      prediction_output = loaded_model(sample_input_tensor) # Output probabilities

    # # Process prediction output
    # prediction_output = prediction_output.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU, convert to numpy
    # predicted_mask = (prediction_output > THRESHOLD).astype(int)

    # print(f"True Labels: {true_label_tensor.cpu().numpy()}")
    # print(f"Predicted Probabilities: {prediction_output.round(2)}") # Display probabilities rounded
    # print(f"Predicted Binary Mask (Threshold={THRESHOLD}): {predicted_mask}")

    # # Map binary mask back to instrument class names
    # predicted_instrument_classes = [
    #     full_dataset.instrument_classes[i] for i, is_present in enumerate(predicted_mask) if is_present
    # ]
    # true_instrument_classes = [
    #     full_dataset.instrument_classes[i] for i, is_present in enumerate(true_label_tensor.cpu().numpy()) if is_present
    # ]

    # print(f"True Instrument Classes: {true_instrument_classes}")
    # print(f"Predicted Instrument Classes: {predicted_instrument_classes}")