"""
Defining common constant variable for training pipeline
"""

PIPELINE_NAME: str = "network_data_training_pipeline"
ARTIFIACT_DIR: str = "artifacts"
FILENAME: str = "PhishingData.csv"
TRAIN_FILENAME: str = "train.csv"
TEST_FILENAME: str = "test.csv"
TARGET_COLUMN: str = "Result"


"""
DATA Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DATABASE_NAME: str = "network_data"
DATA_INGESTION_COLLECTION_NAME: str = "network_metrics"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
