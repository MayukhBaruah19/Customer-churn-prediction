import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Folders to create
folders = [
    'artifacts/',
    'Notebook/',
    'Notebook/data/',
    'src/components/',
    'src/pipeline/',
]

# Files to create
files = [
    'Notebook/EDA.ipynb',
    'Notebook/model_tranning.ipynb',
    'src/__init__.py',
    'src/exception.py',
    'src/logger.py',        
    'src/utils.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'src/components/model_evaluation.py',
    'src/components/model_pusher.py',
    'src/pipeline/__init__.py',
    'src/pipeline/training_pipeline.py',
    'src/pipeline/prediction_pipeline.py',
    'config/config.yaml',
    'params.yaml',
    'schema.yaml',
    'setup.py',
    'app.py',
    'requirements.txt'
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Directory created: {Path(folder).resolve()}")

# Create empty files
for filepath in files:
    path = Path(filepath)
    if (not path.exists()) or (path.stat().st_size == 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        logging.info(f"File created: {path.resolve()}")
    else:
        logging.info(f"File already exists: {path.resolve()}")
