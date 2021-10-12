
import pathlib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

from .GEM import GEMContainer, GEMSimple, GEMText


class MyModel:
    def __init__(self):

        self.model_dir = pathlib.Path(__file__).parent

        self.model = load_model(str(self.model_dir / 'model.h5'))
        self.output_columns = ['predicted_target']
        self.inputs = self.model.input.name
    
    def get_settings(self):
        return {'ig_start_steps': 32,  # 32
                'ig_max_steps': 4096,  # 2048
                'ig_min_error_pct':5.0 # 1.0
               }

    def _transform_input(self, input_df):
        return input_df

    def get_ig_baseline(self, input_df):
        """ This method is used to generate the baseline against which to compare the input. 
            It accepts a pandas DataFrame object containing rows of raw feature vectors that 
            need to be explained (in case e.g. the baseline must be sized according to the explain point).
            Must return a pandas DataFrame that can be consumed by the predict method described earlier.
        """
        return input_df*0

    def predict(self, input_df):
        transformed_input_df = self._transform_input(input_df)
        pred = self.model.predict(transformed_input_df)
        return pd.DataFrame(pred, columns=self.output_columns)
    
    def transform_to_attributable_input(self, input_df):
        """ This method is called by the platform and is responsible for transforming the input dataframe
            to the upstream-most representation of model inputs that belongs to a continuous vector-space.
            For this example, the model inputs themselves meet this requirement.  For models with embedding
            layers (esp. NLP models) the first attributable layer is downstream of that.
        """
        transformed_input = self._transform_input(input_df)

        return {self.inputs: input_df.values}
    
    def compute_gradients(self, attributable_input):
        """ This method computes gradients of the model output wrt to the differentiable input. 
            If there are embeddings, the attributable_input should be the output of the embedding 
            layer. In the backend, this method receives the output of the transform_to_attributable_input() 
            method. This must return an array of dictionaries, where each entry of the array is the attribution 
            for an output. As in the example provided, in case of single output models, this is an array with 
            single entry. For the dictionary, the key is the name of the input layer and the values are the 
            attributions.
        """
        gradients_by_output = []
        attributable_input_tensor = {k: tf.identity(v) for k, v in attributable_input.items()}
        gradients_dic_tf = self._gradients_input(attributable_input_tensor)
        gradients_dic_numpy = dict([key, np.asarray(value)] for key, value in gradients_dic_tf.items()) 
        gradients_by_output.append(gradients_dic_numpy)
        return gradients_by_output    
    
    def _gradients_input(self, x):
        """
        Function to Compute gradients.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self.model(x)

        grads = tape.gradient(preds, x)

        return grads


    def project_attributions(self, input_df, attributions):
        att = []
        for ind, col in enumerate(input_df.columns):
            val = input_df[col].values[0]
            att.append(GEMSimple(display_name=col, feature_name=col,
                                 value=float(val),
                                 attribution=attributions[0][self.inputs][ind]))

        gem_container = GEMContainer(contents=att)

        explanations_by_output = {self.output_columns[0]: gem_container.render()}

        return explanations_by_output


def get_model():
    model = MyModel()
    return model
