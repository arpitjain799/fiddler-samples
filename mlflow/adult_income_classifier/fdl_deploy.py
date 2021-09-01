import logging
from pathlib import Path
  
import pandas as pd

import fiddler.fiddler_api as fdl
from fiddler.core_objects import ModelInputType, ModelTask, MLFlowParams, DeploymentOptions


FIDDLER_ENDPOINT = 'https://your-url.fiddler.ai'
FIDDLER_TOKEN = 'TOKEN_HERE'
ORG_ID = 'ORG_ID'

PROJECT_ID = 'mlflowproject'
DATASET_ID = 'income'
MODEL_ID = 'catboost'

BASELINE_CSV = 'adult_income_train.csv'
TEST_CSV = 'adult_income_test.csv'

def test(client):
    df = pd.read_csv(TEST_CSV)
    df.income = df.income.map(
        {
            '<=50K' : 0,
            '>50K' : 1
        }
    )
    result = client.run_model(PROJECT_ID, MODEL_ID, df.sample(100))
    print(result)

def delete(client):
    # delete dataset if it is already available
    client.delete_model(PROJECT_ID, MODEL_ID, delete_pred=True, delete_prod=True)
    client.delete_dataset(PROJECT_ID, DATASET_ID)

def deploy(client):
    if PROJECT_ID not in client.list_projects():
        client.create_project(PROJECT_ID)

    # upload dataset if it is not present
    if DATASET_ID not in client.list_datasets(PROJECT_ID):
        df = pd.read_csv(BASELINE_CSV)

        df.income = df.income.map(
            {
                '<=50K' : 0,
                '>50K' : 1
            }
        )

        schema = fdl.DatasetInfo.from_dataframe(df, max_inferred_cardinality=1000)
        print(schema)

        # upload dataset
        client.upload_dataset(
            project_id=PROJECT_ID,
            dataset={'baseline': df},
            dataset_id=DATASET_ID,
            info=schema,
        )

    dataset_info = client.get_dataset_info(PROJECT_ID, DATASET_ID)

    target = 'income'
    feature_columns = [col.name for col in dataset_info.columns]
    feature_columns.remove(target)
    print(feature_columns)
    output = 'pred_income'

    model_info = fdl.ModelInfo.from_dataset_info(
        dataset_info=dataset_info,
        target=target,
        features=feature_columns,
        outputs=[output], 
        model_task=ModelTask.BINARY_CLASSIFICATION,
        # metadata_cols=[],
        display_name='MLFlow classification model',
        description='This is a catboost classifier that has been containerized using MLflow docker utility',
    )
    print(model_info)
    deployment_options = DeploymentOptions(
        deployment_type='predictor',
        image_uri='amalfiddler/mlflow_catboost_adult_income_proba:latest',
        port=8080,
    )

    result = client.register_model(
      PROJECT_ID,
      MODEL_ID,
      DATASET_ID,
      model_info,
      deployment_options,
    )
    print(result)

def main():
    client = fdl.FiddlerApi(FIDDLER_ENDPOINT, ORG_ID, FIDDLER_TOKEN)
    delete(client)
    deploy(client)
    test(client)


if __name__ == '__main__':
    main()
