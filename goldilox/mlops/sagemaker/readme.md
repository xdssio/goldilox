# SageMaker instructions

TAG=v1

Test training locally

```bash
# training
python goldilox/mlops/sagemaker/program/training.py --training tests/mlops/sagemaker/training \
  --pipeline tests/mlops/sagemaker/pipeline.pkl \
  --model-dir tests/mlops/sagemaker/output/


# inference
glx serve tests/mlops/sagemaker/output/pipeline.pkl
```

```
Test Docker
```bash
glx build tests/mlops/sagemaker/output/pipeline.pkl --name=goldilox-sagemaker
docker run -it -p 5000:5000 -v $(pwd)/models:/home/app/models -v $(pwd)/logs:/home/app/logs -v $(pwd)/sm/test_dir:/opt/ml lakeml
docker run -it -p 5000:5000 -v $(pwd)/models:/home/app/models -v $(pwd)/logs:/home/app/logs -v $(pwd)/sm/test_dir:/opt/ml lakeml /bin/bash

docker run -it -p 5000:5000 -v $(pwd)/models:/home/app/models -v $(pwd)/logs:/home/app/logs -v $(pwd)/sm/test_dir:/opt/ml lakeml python sm/program/training.py

```

Build and upload

```bash
docker build -f sm/Dockerfile -t lakeml .
REPOSITORY=856378554515.dkr.ecr.eu-central-1.amazonaws.com/goldilocks:$TAG
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin $REPOSITORY
docker tag goldilocks:latest $REPOSITORY
docker push $REPOSITORY


```

Clean untagged images

```bash
IMAGES_TO_DELETE=$( aws ecr list-images --region eu-central-1 --repository-name goldilocks --filter "tagStatus=UNTAGGED" --query 'imageIds[*]' --output json )
aws ecr batch-delete-image --region eu-central-1 --repository-name goldilocks --image-ids "$IMAGES_TO_DELETE" || true

```

Delete algorithm

```bash
aws sagemaker delete-algorithm --algorithm-name goldilocks-v2 --region eu-central-1
```

Create algorithm

```bash
aws sagemaker create-algorithm --algorithm-name goldilocks-v2 --region eu-central-1 --algorithm-description "Behaviors and mitre tag classifiers" \
   --training-specification file://sm/program/configuration/algorithm_configuration.json 

```

Create training job

* Change the now time to have a unique value

```bash

NOW=220220211243
aws sagemaker create-training-job --region eu-central-1 --training-job-name "goldilocks-v2-dev-$NOW" \
  --role-arn arn:aws:iam::856378554515:role/sagemaker-role --enable-managed-spot-training \
  --output-data-config "S3OutputPath=s3://cybear-models/goldilocks/v2/dev" \
  --stopping-condition MaxRuntimeInSeconds=600,MaxWaitTimeInSeconds=600 \
  --input-data-config file://sm/program/configuration/training_channels.json --resource-config file://sm/program/configuration/training_resource.json --algorithm-specification file://sm/program/configuration/training_configuration.json


```

docker build -f sm/Dockerfile -t goldilocks . REPOSITORY=856378554515.dkr.ecr.eu-central-1.amazonaws.com/goldilocks:$TAG
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin $REPOSITORY docker tag
goldilocks:latest $REPOSITORY docker push $REPOSITORY

aws sagemaker delete-algorithm --algorithm-name goldilocks-v2 --region eu-central-1

aws sagemaker create-algorithm --algorithm-name goldilocks-v2 --algorithm-description "Behaviors and mitre tag
classifiers" \
--training-specification file://sm/program/configuration/algorithm_configuration.json --region eu-central-1

NOW=04112020152705 aws sagemaker create-training-job --region eu-central-1 --training-job-name "
goldilocks-v2-dev-$NOW" \
--role-arn arn:aws:iam::856378554515:role/sagemaker-role --enable-managed-spot-training \
--output-data-config "S3OutputPath=s3://cybear-models/goldilocks/v2/dev" \
--stopping-condition MaxRuntimeInSeconds=600,MaxWaitTimeInSeconds=600 \
--input-data-config file://sm/program/configuration/training_channels.json
--resource-config file://sm/program/configuration/training_resource.json
--algorithm-specification file://sm/program/configuration/training_configuration.json

