import urllib.request
import zipfile
import os
import shutil

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip"
TARGET_DIR = r"..\data"
ZIP_PATH = os.path.join(TARGET_DIR, "oulad_uci.zip")

def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    if os.path.exists(os.path.join(TARGET_DIR, "oulad.zip")):
        try:
            os.remove(os.path.join(TARGET_DIR, "oulad.zip"))
        except:
            pass

    print(f"Downloading OULAD from UCI: {UCI_URL} -> {ZIP_PATH}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(UCI_URL, headers=headers)
    
    with urllib.request.urlopen(req, timeout=120) as resp, open(ZIP_PATH, "wb") as out_file:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                print(f"Progress: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({percent:.1f}%)", end="\r")
            else:
                print(f"Downloaded: {downloaded / 1e6:.1f} MB", end="\r")
    
    print(f"\nDownload complete ({downloaded / 1e6:.2f} MB). Verifying and extracting...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        names = zip_ref.namelist()
        print(f"Found {len(names)} items in zip archive:")
        for name in names:
            print("  -", name)
        zip_ref.extractall(TARGET_DIR)
        
    print("\nExtraction complete! Files in TARGET_DIR:")
    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(root, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  * {f}: {size_mb:.2f} MB ({fpath})")

if __name__ == "__main__":
    download_and_extract()
