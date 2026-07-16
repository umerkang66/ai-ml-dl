# Iris Flower Classifier with BentoML

This project demonstrates how to build, test, package, and deploy a machine learning model using BentoML and scikit-learn. The model classifies iris flowers into three species based on physical measurements.

## Table of Contents

1. [BentoML Core Concepts](#bentoml-core-concepts)
2. [Project File Structure](#project-file-structure)
3. [Installation and Setup](#installation-and-setup)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [BentoML CLI Reference](#bentoml-cli-reference)

---

## BentoML Core Concepts

BentoML is an open-source framework designed to simplify the process of packaging, serving, and deploying machine learning models into production-ready APIs. Below are the key concepts implemented in this repository.

### 1. Model Store

BentoML provides a local Model Store to manage trained models, versions, and metadata. By storing models in the Model Store, you decouple model training from model serving.
- **Saving a Model:** Use framework-specific helper APIs (such as `bentoml.sklearn.save_model`) to save trained objects directly to the BentoML model registry. This creates a versioned model entry.
- **Retrieving a Model:** Use `bentoml.models.get("model_name:version")` or `bentoml.models.get("model_name:latest")` to retrieve the model reference.

### 2. Service Definition

A BentoML service defines the network interface, resource specifications, and the runtime logic for serving predictions.
- **Service Decorator:** The `@bentoml.service` decorator marks a class as a BentoML service. You can specify resource requirements, such as CPU or GPU limits, inside the decorator parameters.
- **Model Dependencies:** Declare model references as class-level attributes. BentoML automatically registers these models as dependencies.
- **API Endpoints:** Use the `@bentoml.api` decorator on class methods to expose them as HTTP endpoints. Input and output types can be specified using type hints (e.g., numpy arrays, pandas DataFrames, or Pydantic schemas).

### 3. Runners (Offline Testing)

Runners are abstraction layers representing execution units for running model inference. For testing or simple inference outside the service lifecycle, you can call `.to_runner()` on a model reference, call `init_local()`, and invoke the model predict logic locally.

### 4. Bento Build Configuration

A `bentofile.yaml` specifies how a Bento should be packaged. It defines:
- **Service Entrypoint:** The path to the Python file and service class.
- **Files to Include:** A list of patterns specifying which source files are included.
- **Python Dependencies:** PIP packages and versions required by the runtime.
- **System Labels:** Metadata tags, such as ownership and project identifier.

### 5. Bentos and Containers

A Bento is a unified, self-contained distribution format for machine learning services. Running `bentoml build` packages code, models, configurations, and dependencies into a single Bento archive. Once built, a Bento can be containerized into a Docker image automatically using `bentoml containerize`.

---

## Project File Structure

The project contains the following files:

*   [train.py](file:///D:/Workspace/ai-ml-dl/ml-ds-krish-naik/20-bentoml/train.py): A Python script that loads the Iris dataset, trains a Support Vector Machine (SVM) model, and registers it to the local BentoML Model Store.
*   [test.py](file:///D:/Workspace/ai-ml-dl/ml-ds-krish-naik/20-bentoml/test.py): A helper script for testing the registered model offline using the local runner interface.
*   [service.py](file:///D:/Workspace/ai-ml-dl/ml-ds-krish-naik/20-bentoml/service.py): The main service entrypoint that defines the API server class and endpoints using BentoML service decorators.
*   [bentofile.yaml](file:///D:/Workspace/ai-ml-dl/ml-ds-krish-naik/20-bentoml/bentofile.yaml): The build configuration file detailing packaging instructions, labels, and dependencies.
*   [requirements.txt](file:///D:/Workspace/ai-ml-dl/ml-ds-krish-naik/20-bentoml/requirements.txt): Python package requirements needed to run the project locally.

---

## Installation and Setup

### 1. Create a Virtual Environment

Initialize a Python virtual environment to manage project dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Workflow

### Step 1: Train and Save the Model

Run the training script to train the model and save it to the BentoML local Model Store:

```bash
python train.py
```

This registers the model with the name `iris_classifier` and prints the generated version tag (e.g., `iris_classifier:ug63mtebiklwjedf`).

### Step 2: Test the Saved Model

Verify that the model has been stored properly and is functional:

```bash
python test.py
```

This loads the latest saved model and runs a test prediction on a sample data point.

### Step 3: Run the API Service Locally

Start the BentoML development server to serve the API endpoint:

```bash
bentoml serve service.py:IrisClassifierService --reload
```

The service is now running at `http://localhost:3000`. You can send HTTP POST requests to `http://localhost:3000/classify` or access the Swagger UI directly in your browser.

#### Example Request Payload

```json
[
  [5.9, 3.0, 5.1, 1.8]
]
```

### Step 4: Build the Bento Archive

Compile your project code, dependencies, configurations, and models into a single deployment package:

```bash
bentoml build
```

This outputs a unique Bento tag representing the built container-ready archive.

### Step 5: Containerize with Docker

Convert the built Bento archive into a production-ready Docker image:

```bash
bentoml containerize iris_classifier_service:latest
```

This automatically generates a Dockerfile, installs Python dependencies, and compiles a Docker image that can be run on any environment supporting Docker.

---

## BentoML CLI Reference

| Command | Description |
| :--- | :--- |
| `bentoml models list` | List all models saved in the local Model Store. |
| `bentoml models get <tag>` | Inspect metadata of a saved model. |
| `bentoml models delete <tag>` | Delete a model from the local Model Store. |
| `bentoml list` | List all built Bentos in the local store. |
| `bentoml serve <entrypoint>` | Run the service locally in development mode. |
| `bentoml build` | Package the project into a Bento archive. |
| `bentoml containerize <tag>` | Build a Docker image from a Bento package. |
