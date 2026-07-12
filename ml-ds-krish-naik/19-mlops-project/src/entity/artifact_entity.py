from dataclasses import dataclass


@dataclass
class DataIngestionArtifactEntity:
    train_file_path: str
    test_file_path: str
