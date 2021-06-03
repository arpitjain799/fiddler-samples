import logging
from pathlib import Path
  
import pandas as pd

import fiddler.fiddler_api as fdl
from fiddler.core_objects import ModelInputType, ModelTask, MLFlowParams, DeploymentOptions


#FIDDLER_ENDPOINT = 'https://staging1.fiddler.ai'
#FIDDLER_TOKEN = '3haqPK9-3JjP5kxWxHuVFZACpIBlIAgQNjiyCY7vKkY'
#ORG_ID = 'manojtestorg'

FIDDLER_ENDPOINT = 'https://test.fiddler.ai'
FIDDLER_TOKEN = 'D21nDEXNN_2vcgp_JP0KKmp9topBUzz0b68-JbIHFdk'
ORG_ID = 'testorg'

#FIDDLER_ENDPOINT = 'http://localhost:6100'
#FIDDLER_TOKEN = ''
#ORG_ID = 'onebox'

project_id = 'wine_quality'
dataset_id = 'mlflow_wine_6'
model_id = 'mlflow_model_6'

def test(client):
    df = pd.read_csv('wine-quality.csv')
    result = client.run_model(project_id, model_id, df.head())
    print(result)

def delete(client):
    # delete dataset if it is already available
    client.delete_model(project_id, model_id)
    client.delete_dataset(project_id, dataset_id)


def deploy(client):
    if project_id not in client.list_projects():
        client.create_project(project_id)

    if dataset_id not in client.list_datasets(project_id):
        df = pd.read_csv('wine-quality.csv')
        df['quality'] = df['quality'].astype('float')

        schema = fdl.DatasetInfo.from_dataframe(df, max_inferred_cardinality=1000)
        print(schema)


        # split dataset
        train_df = df[:1000]
        test_df = df[1001:]

        # upload dataset
        client.upload_dataset(
            project_id=project_id,
            dataset={'train': train_df, 'test': test_df},
            dataset_id=dataset_id,
            info=schema,
        )

    dataset_info = client.get_dataset_info(project_id, dataset_id)

    target = 'quality'
    feature_columns = [col.name for col in dataset_info.columns]
    feature_columns.remove(target)

    output = 'pred_quality'

    model_info = fdl.ModelInfo.from_dataset_info(
        dataset_info=dataset_info,
        target=target,
        features=feature_columns,
        outputs={output: (3, 9)} ,
        model_task=ModelTask.REGRESSION,
        metadata_cols=[],
        display_name='mlflow model',
        description='this is a sklearn model from tutorial',
    )

    model_info.mlflow_params = MLFlowParams(
        relative_path_to_saved_model='.',
        #live_endpoint='http://host.docker.internal:8080/invocations'
        live_endpoint='http://testorg-wine-quality-mlflow-model-4:8080/invocations'
    )

    #client.delete_model(project_id, model_id, delete_prod=True, delete_pred=True)
    #artifact_path = Path('.')
    #result = client._upload_model_custom(
        #artifact_path,
        #model_info,
        #project_id, 
        #model_id, 
        #[dataset_id], 
        #deployment_type='predictor',
        #image_uri='manojcheenath/fiddler_examples:latest',
        #port=8080,
    #)

    deployment_options = DeploymentOptions(
        deployment_type='predictor',
        image_uri='manojcheenath/fiddler_examples:latest',
        port=8080,
    )

    result = client.register_model(
      project_id,
      model_id,
      dataset_id,
      model_info,
      deployment_options,
    )
    print(result)


def main():
    client = fdl.FiddlerApi(FIDDLER_ENDPOINT, ORG_ID, FIDDLER_TOKEN)
    #delete(client)
    deploy(client)
    #test(client)


if __name__ == '__main__':
    main()
