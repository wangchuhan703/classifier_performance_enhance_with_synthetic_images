import os

# Set the directory you want to start from
rootDir = 'C:/2024_DL/DomainNet/clipart'
subfolders = []

# Loop through all the entries
for entry in os.listdir(rootDir):
    # Create full path
    path = os.path.join(rootDir, entry)
    # Check if it's a directory
    if os.path.isdir(path):
        formatted_entry = entry.replace('_', ' ')
        subfolders.append(formatted_entry)

print(subfolders)