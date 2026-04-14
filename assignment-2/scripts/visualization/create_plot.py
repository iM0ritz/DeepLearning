import re
import matplotlib.pyplot as plt
import math
import os
import sys

def parse_log_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        losses = []
        val_losses = []
        for line in lines:
            if 'loss:' in line:
                loss_match = re.search(r' - loss: (\d+\.\d+)', line)
                val_loss_match = re.search(r' - val_loss: (\d+\.\d+)', line)
                if loss_match and val_loss_match:
                    losses.append(float(loss_match.group(1)))
                    val_losses.append(float(val_loss_match.group(1)))
        
        epochs = list(range(1, len(losses) + 1))
        return epochs, losses, val_losses
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], [], []

def plot_all_logs(file_paths):
    num_files = len(file_paths)
    if num_files == 0: return

    cols = 2 if num_files > 1 else 1
    rows = math.ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
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
        ax.grid(True, linestyle=':', alpha=0.6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig('loss_plots.png')

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print(f"Usage: python {sys.argv[0]} file1.out file2.out ...")
    else:
        plot_all_logs(files)