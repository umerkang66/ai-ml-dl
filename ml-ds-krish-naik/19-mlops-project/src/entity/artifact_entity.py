from dataclasses import dataclass


@dataclass
class DataIngestionArtifactEntity:
    training_file_path: str
    testing_file_path: str
