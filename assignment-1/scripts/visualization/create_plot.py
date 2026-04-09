import re
import matplotlib.pyplot as plt
import sys
import os
import math

def parse_log_file(file_path):
    """Parses training and validation loss from a Keras-style log file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        epochs = re.findall(r'Epoch (\d+)/', content)
        losses = re.findall(r' - loss: (\d+\.\d+)', content)
        val_losses = re.findall(r' - val_loss: (\d+\.\d+)', content)

        # Convert to numeric
        epochs = [int(e) for e in epochs]
        losses = [float(l) for l in losses]
        val_losses = [float(vl) for vl in val_losses]
        
        return epochs, losses, val_losses
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return [], [], []

def plot_all_logs(file_paths):
    num_files = len(file_paths)
    if num_files == 0:
        print("No files to plot.")
        return

    # Calculate grid dimensions: max 2 columns
    cols = 2 if num_files > 1 else 1
    rows = math.ceil(num_files / cols)

    # Create the figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, path in enumerate(file_paths):
        ax = axes_flat[i]
        epochs, losses, val_losses = parse_log_file(path)
        
        if not epochs:
            ax.set_title(f"Error: {os.path.basename(path)}")
            continue
            
        label_name = os.path.basename(path)
        
        ax.plot(epochs, losses, label='Train Loss', linestyle='--', color='blue')
        ax.plot(epochs, val_losses, label='Val Loss', linewidth=2, color='orange')
        
        ax.set_title(f"Log: {label_name}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots (if num_files is odd)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print(f"Usage: python {sys.argv[0]} file1.out file2.out ...")
    else:
        plot_all_logs(files)