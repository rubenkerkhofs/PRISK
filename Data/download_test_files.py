# download_test_data.py
import os
import requests


def download_files(destination_folder="tests/data"):
    """
    Download external data files from S3 and save them to the `destination_folder`.
    Adjust URLs or file names as needed.
    """
    base_url = "https://kuleuven-prisk.s3.eu-central-1.amazonaws.com/"
    files_to_download = [
        "power.xlsx",
        "hybas_as_lev06_v1c.shp",
        "hybas_as_lev06_v1c.shx",
        "hybas_as_lev06_v1c.dbf",
        "hybas_as_lev06_v1c.prj",
        "HA_L6_outlets_India_constrained.csv",
        "damage_curves.xlsx",
        "Indian_firms.xlsx",
    ]

    os.makedirs(destination_folder, exist_ok=True)

    for file_name in files_to_download:
        file_url = base_url + file_name
        local_path = os.path.join(destination_folder, file_name)
        print(f"Downloading {file_url} -> {local_path}")

        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    download_files()
