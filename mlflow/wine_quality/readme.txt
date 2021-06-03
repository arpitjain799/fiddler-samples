

Train a model in mlflow:
   mlflow run -P alpha=0.5 .

model will be trained and saved to:
   mlruns/0/<run-id>/artifacts/model/

build a docker image from this model artifact:
   mlflow models build-docker -m runs:/<run-id>/model

This will create docker image: 'mlflow-pyfunc-servable'

tag this image eg:
docker tag mlflow-pyfunc-servable:latest manojcheenath/fiddler_examples

upload this image to your registry
docker push manojcheenath/fiddler_examples

run locally
docker run --network=bridge -p 8080:8080 -e host=0.0.0.0 manojcheenath/fiddler_examples


test locally
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://localhost:8080/invocations


