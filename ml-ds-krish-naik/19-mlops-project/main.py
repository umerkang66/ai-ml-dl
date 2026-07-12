from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()

    config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)

    data_ingestion = DataIngestion(data_ingestion_config=config)
    data_ingestion.initiate_data_ingestion()
