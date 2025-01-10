#!/bin/bash

# Array of URLs to download
urls=(
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip"
    "https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip"
)

# Directory to save the downloaded files
output_dir="$WORK/MIC/data/gta"
mkdir -p "$output_dir"

# Loop through each URL, download, unzip, and delete the ZIP file
for url in "${urls[@]}"; do
    # Extract file name from URL
    filename=$(basename "$url")
    
    # Download the ZIP file
    wget -P "$output_dir" "$url"

    # Unzip the file into the output directory
    unzip "$output_dir/$filename" -d "$output_dir"

    # Delete the ZIP file
    rm "$output_dir/$filename"
    
    echo "Finished processing $filename"
    echo "--------------------------------------"
done

echo "All files have been downloaded, extracted, and cleaned up in the $output_dir directory."
