
# Working with MLFlow classification models

This example shows how to create a docker contaienr that can serve model predictions via an endpoint. For purpose of this example we train a classifier model on the [Adult income dataset](https://archive.ics.uci.edu/ml/datasets/adult)

1. Install dependencies

   ```pip install -r requirements.txt```

2. Train a model and build a docker conatiner, by running:

   ```python classifier_tutorial.py --docker_name <docker-image-tag>``` 
   
   This will train the model and it will be saved to:

   ``` mlruns/0/<run-id>/artifacts/model/ ```

   Additionally, a docker image will also be created that can then be pushed to a registry

3. Upload this image to a docker registry, which is accessable to Fiddler

    ``` docker push amalfiddler/mlflow_catboost_adult_income_proba```

4. Run and test this image using

    ``` docker run --network=bridge -p 8080:8080 -e host=0.0.0.0 amalfiddler/mlflow_catboost_adult_income_proba```

    To test this docker image locally:

    ``` python test_model_container.py --url http://localhost:8080/invocations --csv adult_income_test.csv --target income```

9. You can now upload this model to fiddler by calling register_model:

   ```
    deployment_options = DeploymentOptions(
        deployment_type='predictor',
        image_uri='amalfiddler/mlflow_catboost_adult_income_proba:latest',
        port=8080,
    )
   ```

   ```
   result = client.register_model(
      project_id,
      model_id,
      dataset_id,
      model_info,
      deployment_options,
    )
   ```

   Check out the full example here [fdl_deploy.py](./fdl_deploy.py)
