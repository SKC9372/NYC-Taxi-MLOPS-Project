from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# List of files to be created
files = [
    "src/__init__.py",
    "src/logger.py",
    "src/data/__init__.py",
    "src/data/extract_dataset.py",
    "src/data/make_dataset.py",
    "src/features/__init__.py",
    "src/features/build_features.py",
    "src/features/data_preprocessing.py",
    "src/features/distances.py",
    "src/features/modify_features.py",
    "src/features/outliers_removal.py",
    "src/models/__init__.py",
    "src/models/predict_model.py",
    "src/models/train_model.py",
    "src/visualizations/__init__.py",
    "src/visualizations/plot_results.py",
    "src/visualizations/visualize.py"
]

# Create folders and files
for file in files:
    folder, file_name = os.path.split(file)

    # Create folder if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory created: {folder}")

    # Create file if it doesn't exist
    file_path = Path(folder) / file_name
    if not file_path.exists():
        file_path.touch()
        logging.info(f"File created: {file_path}")
    else:
        logging.warning(f"File already exists: {file_path}")
