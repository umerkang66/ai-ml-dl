# End-to-End Machine Learning Starter Project - Master Agent Prompt

This file contains an analysis of the End-to-End Machine Learning starter project structure and a complete, robust prompt designed for AI coding agents (such as **Antigravity / AGY**, **Claude Code**, **Cursor**, **ChatGPT**, etc.). 

When you provide the prompt below to any AI coding agent, it will recreate the exact project structure, setup scripts, requirement files, logging, custom exception handling, component modules, and pipeline starter files complete with explanatory comments.

---

## 📐 Project Structure Summary

```text
18-end-to-end-project-1/
├── .gitignore                      # Git ignore patterns for Python, venv, models, logs
├── README.md                       # Project title & overview
├── notes.txt                       # Setup command reference (conda environment setup)
├── requirements.txt                # Required Python packages & editable install trigger (-e .)
├── setup.py                        # Package metadata & automatic dependency installation
└── src/                            # Source code package directory
    ├── __init__.py                 # Marks 'src' as a Python package
    ├── logger.py                   # Custom timestamped logging module
    ├── exception.py                # Custom exception handling capturing file & line numbers
    ├── utils.py                    # Shared utility functions module (placeholder)
    ├── components/                 # Core ML processing components
    │   ├── __init__.py             # Marks 'components' as a package
    │   ├── data_ingestion.py       # Data fetching, splitting, and storage component
    │   ├── data_transformation.py  # Feature engineering & preprocessing pipeline component
    │   └── model_trainer.py        # Model training, evaluation & hyperparameter tuning component
    └── pipeline/                   # Execution workflows
        ├── __init__.py             # Marks 'pipeline' as a package
        ├── train_pipeline.py       # Orchestrates the model training workflow
        └── predict_pipeline.py     # Inference pipeline for single/batch predictions
```

---

## 🤖 AI Agent Prompt

Copy and paste the prompt block below into any AI Coding Agent (Antigravity, AGY, Claude Code, Cursor, etc.) to recreate this starter project in a new directory.

```markdown
You are an expert Machine Learning & Software Engineer. Your task is to initialize a clean, modular, production-ready End-to-End Machine Learning starter project structure with all necessary files, package configurations, custom logging, exception handling, and component placeholders.

Follow the instructions below to create every folder and file with the exact structure and detailed explanatory comments.

---

### Project Architecture & Directory Structure

Create the following directory hierarchy in the project root:

```text
.
├── .gitignore
├── README.md
├── notes.txt
├── requirements.txt
├── setup.py
└── src/
    ├── __init__.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    └── pipeline/
        ├── __init__.py
        ├── predict_pipeline.py
        └── train_pipeline.py
```

---

### File Specification & Starter Code

Create each of the following files with the exact specified content and explanatory comments:

#### 1. `setup.py`
This file allows the project to be installed as a local Python package (`mlproject`). It automatically parses dependencies from `requirements.txt` and excludes `-e .` to avoid recursion.

```python
from setuptools import find_packages, setup

# Special flag used in requirements.txt to trigger local package installation in editable mode
HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list[str]:
    """
    Reads the requirements.txt file and returns a list of dependencies.
    Removes newline characters and filters out the '-e .' editable package flag.
    """
    with open(file_path) as file:
        requirements = file.readlines()
        # Strip trailing newlines from each requirement line
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' if present so setuptools doesn't attempt to install it as a external package
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Metadata and setup configuration for the local ML package
setup(
    name="mlproject",
    version="0.0.1",
    author="Muhammad Umer",
    author_email="ugulzar4512@gmail.com",
    packages=find_packages(),  # Automatically detects packages with __init__.py under src/
    install_requires=get_requirements("requirements.txt"),
)
```

---

#### 2. `requirements.txt`
Specifies required libraries. Including `-e .` at the end automatically triggers `setup.py` when running `pip install -r requirements.txt`.

```text
pandas
numpy
seaborn
matplotlib
-e .
```

---

#### 3. `notes.txt`
Quick reference notes for environment setup.

```text
conda create --prefix ./venv python=3.12.7
```

---

#### 4. `README.md`
Project documentation template.

```markdown
# 18-end-to-end-project-1

End-to-End Machine Learning Modular Starter Project.
```

---

#### 5. `.gitignore`
Standard ignore configuration for virtual environments, build artifacts, model checkpoints, and logs.

```gitignore
# --- User-specified ---
venv
*-info

# --- Python ---
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
*.egg
pip-log.txt
pip-delete-this-directory.txt

# --- Virtual environments ---
env/
.venv/
env.bak/
venv.bak/
ENV/

# --- Jupyter Notebook ---
.ipynb_checkpoints/
profile_default/
ipython_config.py

# --- Environment / secrets ---
.env
.env.*
*.cfg
!setup.cfg
secrets.yaml
config_local.py

# --- ML / Data Science specific ---
data/
datasets/
*.csv
*.tsv
*.parquet
*.h5
*.hdf5
*.feather

# Model artifacts / checkpoints
*.pt
*.pth
*.ckpt
*.onnx
*.pb
*.h5model
models/
checkpoints/
saved_models/
weights/

# Logs & experiment tracking
logs/
*.log
runs/
lightning_logs/
mlruns/
wandb/
tensorboard/
events.out.tfevents.*

# --- IDE / Editor ---
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# --- OS files ---
Thumbs.db
desktop.ini

# --- Testing / Coverage ---
.pytest_cache/
.coverage
htmlcov/
.tox/
```

---

#### 6. Package Initializers (`__init__.py`)
Create empty `__init__.py` files in the following locations to make them Python packages:
- `src/__init__.py`
- `src/components/__init__.py`
- `src/pipeline/__init__.py`

---

#### 7. `src/logger.py`
Configures a centralized logging utility that creates a new timestamped log file inside a `logs/` directory upon execution.

```python
import logging
import os
from datetime import datetime

# Generate log filename based on current timestamp (YYYY-MM-DD_HH-MM-SS.log)
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Define the log directory path under the project root
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Ensure the logs directory exists
os.makedirs(logs_path, exist_ok=True)

# Full path to the specific log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure standard logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

---

#### 8. `src/exception.py`
Provides detailed exception tracking by extracting the script name, line number, and error message using Python's `sys` module.

```python
import sys


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information including file name, line number, and error message.
    """
    _, _, exc_info = error_detail.exc_info()
    file_name = exc_info.tb_frame.f_code.co_filename
    line_number = exc_info.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error}]"

    return error_message


class CustomException(Exception):
    """
    Custom exception class that formats and prints detailed traceback info.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
```

---

#### 9. `src/utils.py`
Helper functions module for shared tasks like saving objects, loading models, evaluating models, etc.

```python
"""
utils.py: Helper functions module for shared utilities across the ML project.
Contains functionality such as pickle dump/load operations, metric calculations, and database connectors.
"""

import os
import sys
# Import custom logger and exception handler when needed:
# from src.logger import logging
# from src.exception import CustomException
```

---

#### 10. `src/components/data_ingestion.py`
Data Ingestion component placeholder for fetching data from sources (databases, APIs, CSVs) and splitting into train/test sets.

```python
"""
data_ingestion.py: Data Ingestion Component.
Responsible for reading raw data from various sources (database, remote storage, local CSVs),
performing train-test splits, and saving raw artifacts into the data directory.
"""

import os
import sys
# from src.logger import logging
# from src.exception import CustomException
```

---

#### 11. `src/components/data_transformation.py`
Data Transformation component placeholder for handling missing values, scaling, encoding, and feature engineering pipelines.

```python
"""
data_transformation.py: Data Transformation Component.
Responsible for feature engineering, handling missing values, standard scaling, categorical encoding,
and saving the preprocessing pipeline object (e.g., ColumnTransformer / Pipeline pkl).
"""

import os
import sys
# from src.logger import logging
# from src.exception import CustomException
```

---

#### 12. `src/components/model_trainer.py`
Model Trainer component placeholder for training ML algorithms, hyperparameter tuning, and selecting the best performing model.

```python
"""
model_trainer.py: Model Trainer Component.
Responsible for training various machine learning models, evaluating metrics (R2, RMSE, Accuracy),
hyperparameter optimization, and serializing the best model.
"""

import os
import sys
# from src.logger import logging
# from src.exception import CustomException
```

---

#### 13. `src/pipeline/train_pipeline.py`
Training Pipeline placeholder to trigger the end-to-end ingestion, transformation, and model training workflow sequentially.

```python
"""
train_pipeline.py: Training Pipeline Runner.
Orchestrates the sequential execution of Data Ingestion -> Data Transformation -> Model Training.
"""

import os
import sys
# from src.logger import logging
# from src.exception import CustomException
```

---

#### 14. `src/pipeline/predict_pipeline.py`
Prediction Pipeline placeholder for accepting new input data, applying saved transformations, and generating model predictions.

```python
"""
predict_pipeline.py: Prediction Pipeline.
Loads saved preprocessing artifacts and trained model to make predictions on new input data.
Useful for Web APIs (Flask/FastAPI/Streamlit) or batch prediction jobs.
"""

import os
import sys
# from src.logger import logging
# from src.exception import CustomException
```

---

### Verification Step

After creating all directories and files:
1. Create a virtual environment using `python -m venv venv` or `conda create --prefix ./venv python=3.12.7`.
2. Activate the environment.
3. Run `pip install -r requirements.txt` to verify that dependencies install smoothly and the local package `mlproject` is installed in editable mode (`-e .`).
4. Confirm `mlproject.egg-info` is generated in the root directory.
```
