import pickle
import pandas as pd
import os

class SyntheticDataSampler:
    def __init__(self):
        """
        Initializes the SyntheticDataSampler by loading the fixed pickle model.
        """
        self.pickle_path = os.path.join(os.path.dirname(__file__), "synth_model.pkl")
        if not os.path.exists(self.pickle_path):
            raise FileNotFoundError("The model file 'synth_model.pkl' is missing in the package.")
        
        with open(self.pickle_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def sample(self, num_rows: int = 500) -> pd.DataFrame:
        """
        Generates synthetic data by sampling from the loaded model.

        :param num_rows: Number of rows to sample (default is 500).
        :return: A pandas DataFrame containing the sampled data.
        """
        return self.model.sample(num_rows)
    
    def save_sample(self, num_rows: int, output_path: str):
        """
        Samples data and saves it to a CSV file.

        :param num_rows: Number of rows to sample.
        :param output_path: Path to save the sampled data as a CSV file.
        """
        df = self.sample(num_rows)
        df.to_csv(output_path, index=False)
        print(f"Sampled data saved to {output_path}")
