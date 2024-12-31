# download_test_data.py
import os
import requests
import zipfile


def download_files(destination_folder="Data/app"):
    """
    Download external data files from S3, save them to the `destination_folder`,
    and handle additional processing (e.g., unzip files).
    """
    base_url = "https://kuleuven-prisk.s3.eu-central-1.amazonaws.com/"
    files_to_download = [
        # Shared
        "power.xlsx",
        "damage_curves.xlsx",
        # India
        "hybas_as_lev06_v1c.shp",
        "hybas_as_lev06_v1c.shx",
        "hybas_as_lev06_v1c.dbf",
        "hybas_as_lev06_v1c.prj",
        "HA_L6_outlets_India_constrained.csv",
        "Indian_firms.xlsx",
        # Thailand
        "financial_data.xlsx",
        "lev06_outlets_final_clipped_Thailand_no_duplicates.csv",
        # Random numbers (zip file)
        "random-numbers/prisk-random-numbers.zip",
    ]

    os.makedirs(destination_folder, exist_ok=True)

    for file_name in files_to_download:
        file_url = base_url + file_name
        local_path = os.path.join(destination_folder, os.path.basename(file_name))
        print(f"Downloading {file_url} -> {local_path}")

        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # If the file is the zip file, extract its contents and delete the zip
        if file_name.endswith(".zip"):
            unzip_file(local_path, destination_folder)


def unzip_file(zip_path, extract_to):
    """
    Unzip the specified file into the given folder and remove the zip file.
    """
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    print(f"Deleted zip file: {zip_path}")


if __name__ == "__main__":
    download_files()
