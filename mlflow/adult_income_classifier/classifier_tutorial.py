from sys import version_info 
import cloudpickle
import pandas as pd
import argparse

import mlflow
import mlflow.pyfunc
from mlflow.models import Model
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
import pandas as pd
import os
import numpy as np
np.set_printoptions(precision=4)
import catboost
from catboost import *
from catboost import datasets
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from mlflow.models.flavor_backend_registry import get_flavor_backend

print(f'Catboost version: {catboost.__version__}')

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major,
    minor=version_info.minor,
    micro=version_info.micro
)

def get_dataset(type='adult'):
    """Returns train and test splits"""
    if type == 'amazon':
        (train_df, test_df) = catboost.datasets.amazon()
        train_df.to_csv('amazon_train.csv')
        test_df.to_csv('amazon_test.csv')
        y = train_df.ACTION
        X = train_df.drop('ACTION', axis=1)
        cat_features = list(range(0, X.shape[1]))
    else:
        # (train_df, test_df) = catboost.datasets.adult()
        # train_df.to_csv('adult_income_train.csv')
        # test_df.to_csv('adult_income_test.csv')
        train_df = pd.read_csv('adult_income_train.csv', index_col=0)
        test_df = pd.read_csv('adult_income_test.csv', index_col=0)
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        train_df.income = train_df.income.map(
            {
                '<=50K' : 0,
                '>50K' : 1
            }
        )
        y = train_df.income
        X = train_df.drop('income', axis=1)
        cat_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex',
                        'native-country']
        print(cat_features)

    print('Training data snippet:')
    print(X.head())

    X_train, X_validation,\
    y_train, y_validation = train_test_split(X, y,
                    train_size=0.8, random_state=42)
    # print(X_train.sample(1).to_json(orient='split'))
    return (X_train, y_train, X_validation, y_validation), cat_features

def train_and_log(
    dataset,
    cat_features,
    iterations=50,
    learning_rate=0.5,
    depth=3,
    allow_writing_files=False
):
    """Trains a catboost model on the dataset and logs as mlflow project"""
    X_train, y_train, X_validation, y_validation = dataset
    params = {
        "iterations": iterations,
        "learning_rate" : learning_rate,
        "depth": depth,
        "allow_writing_files": allow_writing_files,
    }
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_validation, y_validation),
        verbose=True
    )
    print('Model has been trained: ' + str(model.is_fitted()))
    print('Model params: ')
    print(model.get_params())

    print('Logging trained catboost model with MLFlow')
    # for inferring signature
    from mlflow.models.signature import infer_signature
    print(model.predict_proba(X_train.sample(1)))
    signature = infer_signature(X_train, model.predict(X_train))
    print(signature)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.catboost.log_model(model, artifact_path="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")
        print(f'Model URI: {model_uri}')
    return model, model_uri

def create_docker_image(
    model_uri,
    image_name,
    tag='latest',
    dataset=None
):
    """Creates a docker image with mflow's pyfunc model"""
    # Load mlflow model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Get predictions
    if dataset is not None:
        X_train, y_train, X_validation, y_validation = dataset
        preds = loaded_model.predict(X_validation)
        # print("predictions:", preds)
        print("Successfuly laoded the model as MLFLOW pyfunc type")

    # docker image creation
    with TempDir() as tmp:
        model_path = _download_artifact_from_uri(model_uri+'/MLmodel', output_path=tmp.path())
        flavor_name, flavor_backend = get_flavor_backend(Model.load(model_path), build_docker=True)
        print(flavor_name)
        mlflow_home = os.environ.get("MLFLOW_HOME", None)

    flavor_backend.build_image(
        model_uri,
        image_name+':'+tag,
        mlflow_home=mlflow_home,
        install_mlflow=True
    )

class CatBoostClassifierScorer(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(context.artifacts['model_binary_path'])

    def predict(self, context, model_input):
        # this call returns probability for both negative and positive class
        scores = self.catboost_model.predict_proba(model_input)
        # since this is a binary classifier select one of the classes
        # we select the positive class which is income>50k for this example
        return scores[:, 1]

def get_conda_env():
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            'pip',
            {'pip': [
                'mlflow',
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'catboost=={}'.format(catboost.__version__)
                ]
            }
        ],
        'name': 'catboost_env'
    }
    return conda_env

def save_model_pyfunc(
    python_model,
    model_binary_path, 
    model_name='catboost_docker_tutorial'
):
    # get conda environment info
    conda_env = get_conda_env()

    # specify location of artifacts
    artifacts = {
        'model_binary_path': model_binary_path 
    }
    # save the model as pyfunc
    with mlflow.start_run(run_name="catboost tutorial") as run:
        model_path = f'{model_name}-{run.info.run_uuid}'
        mlflow.pyfunc.save_model(
            path=model_path,
            python_model=python_model,
            artifacts=artifacts,
            conda_env=conda_env,
            code_path=None, # way to add custom code
        )
    return model_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--docker_tag', help='Name of the docker tag', type=str,
        default='amalfiddler/mlflow_catboost_adult_income_proba'
    )
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()

    print('\nDownloading training data...')
    dataset, cat_features = get_dataset()

    print('\nTraining model...')
    model_binary_path = 'catboost_classifier.cb'
    model_obj, _ = train_and_log(dataset, cat_features=cat_features)
    model_obj.save_model(model_binary_path)
    
    print('\nSaving model in MLFlow docker utility compatible format')
    model_path = save_model_pyfunc(
        python_model=CatBoostClassifierScorer(),
        model_binary_path=model_binary_path
    )
    
    print('\nCreating docker image...')
    create_docker_image(
        model_uri=model_path,
        image_name=args.docker_tag,
        dataset=dataset
    )
    