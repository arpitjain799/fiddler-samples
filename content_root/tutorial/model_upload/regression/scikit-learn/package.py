import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

PACKAGE_PATH = Path(__file__).parent

OUTPUT_COLUMN = ['predicted_quality']

class FiddlerModel:

    def __init__(self):
        
        # Load the model
        with open(PACKAGE_PATH / 'model.pkl', 'rb') as pkl_file:
            self.model = pickle.load(pkl_file)

    def predict(self, input_df):
        
        # Store predictions in a DataFrame
        return pd.DataFrame(self.model.predict(input_df), columns=OUTPUT_COLUMNS)

def get_model():
    return FiddlerModel()