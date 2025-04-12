"""
Rename files in dir '/rename_files' (place your images here)
use: python3 rename.py [NAME] [DESCRIPTION 1] [opt. DESCRIPTION 2]
example: python3 rename.py "avocado" "raw"
    -> rename_files/avocado_raw_1.jpg
tip: make sure to navigate to this directory before running the script:
    cd datasets/image_preprocessing
"""

import os
import sys

def main():
    # usage
    if len(sys.argv) < 3:
        print("Usage: python3 rename.py [NAME] [DESCRIPTION 1] [opt. DESCRIPTION 2]")
        return
    # get the name and descriptions list from args
    name = sys.argv[1]
    descriptions = sys.argv[2:]
    joined_desc = "_".join(descriptions)

    # UPPERCASE 
    name = name.upper()
    joined_desc = joined_desc.upper()

    dir_path = "./rename_files"

    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return
    
    # sort files in dir
    files = sorted(os.listdir(dir_path))

    count = 1
    extensions = [".jpg", ".jpeg", ".png"]

    # for each file in the directory, rename it using the file naming format
    for filename in files:
        file_path = os.path.join(dir_path, filename)

        extension = os.path.splitext(filename)[1].lower()
        # check if the file is an image
        if extension not in extensions:
            print(f"Skipping {filename}: not a valid image file.")
            continue

        # rename the file
        new_name = f"{name}_{joined_desc}_{count}{extension}"
        new_path = os.path.join(dir_path, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed {filename} -> {new_name}")
        count += 1
        
    print(f"Renamed {count - 1} files.")


if __name__ == "__main__":
    main()