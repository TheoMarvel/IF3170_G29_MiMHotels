import os
import pickle

class Serializer: # Save and load models
    @staticmethod
    def save(model, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)