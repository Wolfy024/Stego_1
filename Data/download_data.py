import os
import zipfile
import subprocess
import shutil

def setup_kaggle_json(json_path):
    """Place kaggle.json in ~/.kaggle/ so Kaggle CLI can use it"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    dest = os.path.join(kaggle_dir, "kaggle.json")
    shutil.copy(json_path, dest)
    os.chmod(dest, 0o600)  # secure permissions
    print(f"✅ kaggle.json copied to {dest}")

def download_animals10():
    dataset = "alessiocorrado99/animals10"
    output_dir = os.path.join(os.getcwd(), "Animals-10")
    zip_path = os.path.join(output_dir, "animals10.zip")

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Downloading dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset,
        "-p", output_dir
    ], check=True)

    # Find and unzip
    for f in os.listdir(output_dir):
        if f.endswith(".zip"):
            zip_path = os.path.join(output_dir, f)
            break

    print("📂 Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    print(f"✅ Dataset ready in: {output_dir}")


if __name__ == "__main__":
    setup_kaggle_json("Data/kaggle_api_key.json")
    download_animals10()
