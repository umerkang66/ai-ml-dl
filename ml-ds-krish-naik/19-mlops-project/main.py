from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
)
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()

    data_ingestion_config = DataIngestionConfig(
        training_pipeline_config=training_pipeline_config
    )

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation_config = DataValidationConfig(
        training_pipeline_config=training_pipeline_config
    )

    validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config,
    )
    data_validation_artifact = validation.initiate_data_validation()
