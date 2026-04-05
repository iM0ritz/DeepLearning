import sys

def clean_keras_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace carriage returns with standard newlines so we can 
    # separate out all the messy progress bar updates
    lines = content.replace('\r', '\n').split('\n')

    for line in lines:
        line = line.strip()
        
        # Keep only the Epoch headers and the final summary line
        if line.startswith("Epoch "):
            print(line)
        elif "- val_loss:" in line:
            print(line)

if __name__ == "__main__":
    # Ensure the user provided a filename argument
    if len(sys.argv) > 1:
        clean_keras_log(sys.argv[1])
    else:
        print("Usage: python clean_logs.py <path_to_log_file>")