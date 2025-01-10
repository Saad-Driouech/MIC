import gdown
import zipfile
import os


# Step 1: Set up paths
download_url = "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw"
download_path = "/home/woody/iwnt/iwnt134h/MIC/data/office-home/office-home.zip"
unzip_path = "/home/woody/iwnt/iwnt134h/MIC/data/"

# Ensure the download directory exists
os.makedirs(os.path.dirname(download_path), exist_ok=True)

# Step 2: Download the file using gdown
try:
    print("Downloading file from Google Drive...")
    gdown.download(download_url, download_path, quiet=False)
    print("Download completed successfully.")
except Exception as e:
    print(f"Error during download: {e}")
    exit(1)

# # Step 3: Attempt to unzip the file
# try:
#     print("Extracting the zip file...")
#     with zipfile.ZipFile(download_path, 'r') as zip_ref:
#         zip_ref.extractall(unzip_path)
#     print("Files extracted successfully.")
# except zipfile.BadZipFile:
#     print("Failed to unzip. The downloaded file may be corrupted.")
# except Exception as e:
#     print(f"An error occurred while extracting the zip file: {e}")

# # Cleanup: Optionally remove the zip file after extraction
# try:
#     if os.path.exists(download_path):
#         os.remove(download_path)
#         print("Cleaned up the zip file after extraction.")
# except Exception as e:
#     print(f"Failed to clean up zip file: {e}")
