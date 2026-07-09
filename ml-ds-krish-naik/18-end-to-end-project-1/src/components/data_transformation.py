import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preproecessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for data transformation"""

        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical and categorical pipeline completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target = "math_score"

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_preprocessed = preprocessing_obj.fit_transform(X_train)
            X_test_preprocessed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_preprocessed, np.array(y_train)]
            test_arr = np.c_[X_test_preprocessed, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preproecessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preproecessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
