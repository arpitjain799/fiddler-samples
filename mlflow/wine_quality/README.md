
# Working with MLFlow models

This example shows how to upload a model packaged using MLFlow 

1. Install MLFlow

   Follow instruction here to install **[mlflow](https://www.mlflow.org/docs/latest/quickstart.html)** 

2. Write training code and the usual MLFlow metadata files:

   This example is based on the [sklearn_elasticnet_wine](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine) mlflow example

   - [MLproject](./MLproject)
   - [conda.yaml](./conda.yaml)
   - [train.py](./train.py)
   - [wine-quality.csv](./wine-quality.csv)

   Make sure a model artifact is saved to mlrun by calling, at the end of your training code:

   ``` mlflow.sklearn.log_model(lr, "model") ```

3. Train a model in mlflow, by running:

   ```mlflow run -P alpha=0.5 .i ``` 

   This will train the model and it will be saved to:

   ``` mlruns/0/<run-id>/artifacts/model/ ```

4. We can use this saved model artifact to build a docker image:

   ``` mlflow models build-docker -m runs:/<run-id>/model ```

   This will create docker image: 'mlflow-pyfunc-servable'

5. tag this image, so we can push it to a docker registry, eg:

    ``` docker tag mlflow-pyfunc-servable:latest manojcheenath/fiddler_examples```

6. upload this image to a docker registry, which is accessable to Fiddler

    ``` docker push manojcheenath/fiddler_examples```

7. You can run this docker container locally

    ``` docker run --network=bridge -p 8080:8080 -e host=0.0.0.0 manojcheenath/fiddler_examples```

8. To test this docker image locally:

    ``` curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://localhost:8080/invocations```

9. You can now upload this model to fiddler by calling register_model:

   ```
    deployment_options = DeploymentOptions(
        deployment_type='predictor',
        image_uri='manojcheenath/fiddler_examples:latest',
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


