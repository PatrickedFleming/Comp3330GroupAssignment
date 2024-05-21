import os

def rename_files_in_folders(directory):
    # Loop through each folder in the directory
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        
        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            # Get a list of files in the folder
            files = os.listdir(folder_path)
            
            # Initialize increment counter
            increment = 1
            
            # Loop through each file in the folder
            for filename in files:
                
                # Construct the new filename
                new_filename = f"{foldername}_{increment}.jpg"
                
                # Append increment to the filename if the new filename already exists
                while os.path.exists(os.path.join(folder_path, new_filename)):
                    increment += 1
                    new_filename = f"{foldername}_{increment}.jpg"
                
                # Rename the file
                os.rename(
                    os.path.join(folder_path, filename),
                    os.path.join(folder_path, new_filename)
                )
                
                # Increment counter
                increment += 1

# Replace 'directory_path' with the path to your directory
directory_path = 'dataset'
rename_files_in_folders(directory_path)