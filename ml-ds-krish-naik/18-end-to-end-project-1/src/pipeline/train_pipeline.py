import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started.")

            # 1. Data Ingestion
            logging.info("Initiating Data Ingestion in training pipeline.")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            logging.info("Initiating Data Transformation in training pipeline.")
            data_transformation = DataTransformation()
            train_data, test_data, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_data_path, test_data_path
                )
            )

            # 3. Model Training
            logging.info("Initiating Model Training in training pipeline.")
            trainer = ModelTrainer()
            r2_square = trainer.initiate_model_trainer(train_data, test_data)

            logging.info(
                f"Training pipeline completed successfully. Model R2 Score: {r2_square}"
            )
            return r2_square

        except Exception as e:
            logging.error("Exception occurred in training pipeline run.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    r2_score = pipeline.run_pipeline()
    print(f"Model Training Completed. R2 Score: {r2_score}")
