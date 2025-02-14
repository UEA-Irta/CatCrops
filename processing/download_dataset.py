# -*- coding: utf-8 -*-
"""
ZIP File Downloader & Extractor

This script allows you to:
1. Download a ZIP file from a given URL.
2. Save the downloaded file to a specified path.
3. Extract its contents to a target folder.
4. Optionally, delete the ZIP file after extraction.

Usage:
Run the script from the terminal with the following command:
    python download_dataset.py --url "https://www.kaggle.com/api/v1/datasets/download/irtaremotesensing/catcrops-dataset" --zip_path "catcrops_dataset.zip" --extract_folder "./"

Arguments:
    --url             URL of the ZIP file to download.
    --zip_path        Path where the downloaded ZIP file will be saved.
    --extract_folder  Folder where the ZIP file will be extracted.
    --keep_zip        Keep the ZIP file after extraction.

author:
datetime:14/2/2025 14:42
"""

import requests
import zipfile
import os
import argparse
from tqdm import tqdm  # Import tqdm for progress bar

# 🛠 Parse command-line arguments
parser = argparse.ArgumentParser(description="Download and extract a ZIP file.")
parser.add_argument("--url", type=str, required=True, help="URL of the ZIP file to download")
parser.add_argument("--zip_path", type=str, required=True, help="Path to save the downloaded ZIP file")
parser.add_argument("--extract_folder", type=str, required=True, help="Folder where the ZIP file will be extracted")
parser.add_argument("--keep_zip", action="store_true",
                    help="Keep the ZIP file after extraction. If this flag is not provided, "
                         "the ZIP file will be deleted by default.")

args = parser.parse_args()

# Assign arguments to variables
url = args.url
zip_path = args.zip_path
extract_folder = args.extract_folder
keep_zip = args.keep_zip

# Download the ZIP file with a progress bar
print(f"\nDownloading ZIP file from: {url}")

# Get the total file size from the server
response = requests.get(url, stream=True)
total_size = int(response.headers.get("content-length", 0))

# Open file and write in chunks while updating the progress bar
with open(zip_path, "wb") as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
) as progress:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
        progress.update(len(chunk))

print("Download completed!")

# Create the extraction folder if it does not exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Extract the ZIP file with a progress bar
print(f"\n Extracting files to: {extract_folder}")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    file_list = zip_ref.namelist()  # List of files in ZIP
    with tqdm(total=len(file_list), desc="Extracting", unit="file") as progress:
        for file in file_list:
            zip_ref.extract(file, extract_folder)
            progress.update(1)

print("Extraction completed!")

# Delete the ZIP file after extraction
if not keep_zip:
    os.remove(zip_path)
    print(f"\n️ File {zip_path} deleted.")
print("=" * 60)
print("Process completed successfully!")
print("=" * 60)
