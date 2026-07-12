from datetime import datetime
import os
from src.constants import training_pipeline_constants


class TrainingPipelineConfig:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.pipeline_name = training_pipeline_constants.PIPELINE_NAME
        self.artifact_name = training_pipeline_constants.ARTIFIACT_DIR
        self.artifact_dir = os.path.join(
            training_pipeline_constants.ARTIFIACT_DIR, self.timestamp
        )


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline_constants.FILENAME,
        )
        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_INGESTED_DIR,
            training_pipeline_constants.TRAIN_FILENAME,
        )
        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline_constants.DATA_INGESTION_INGESTED_DIR,
            training_pipeline_constants.TEST_FILENAME,
        )
        self.train_test_ratio = (
            training_pipeline_constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name = (
            training_pipeline_constants.DATA_INGESTION_COLLECTION_NAME
        )
        self.database_name = training_pipeline_constants.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline_constants.DATA_VALIDATION_DIR_NAME,
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_VALID_DIR,
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_INVALID_DIR,
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline_constants.TRAIN_FILENAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline_constants.TEST_FILENAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline_constants.TRAIN_FILENAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline_constants.TEST_FILENAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
