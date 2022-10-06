import argparse
import logging
import os
from warnings import filterwarnings

from goldilox import Pipeline

filterwarnings('ignore')

logger = logging.getLogger()

SM_CHANNEL_TRAIN = os.environ.get('SM_TRAIN_CHANNEL', '/opt/ml/input/data/training')
SM_CHANNEL_PIPELINE = os.environ.get('SM_CHANNEL_PIPELINE', '/opt/ml/input/data/pipelines/pipeline.pkl')
HYPERPARAMETERS_PATH = os.environ.get('SM_CHANNEL_AUGMENTATION', '/opt/ml/input/config/hyperparameters.json')
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', '/opt/ml/checkpoints')


def cast(value):
    def isfloat(x):
        try:
            a = float(x)
        except ValueError:
            return False
        else:
            return True

    def isint(x):
        try:
            a = float(x)
            b = int(a)
        except ValueError:
            return False
        else:
            return a == b

    if isinstance(value, list):
        return value
    if isint(value):
        return int(float(value))
    elif isfloat(value):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=str, default=SM_CHANNEL_TRAIN)
    parser.add_argument('--pipeline', type=str, default=SM_CHANNEL_PIPELINE)
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--hyperparameters', type=str, default=HYPERPARAMETERS_PATH)
    parser.add_argument('--checkpoints', type=str, default=CHECKPOINTS_DIR)
    args, _ = parser.parse_known_args()
    training_path = os.path.join(args.training)
    pipeline_path = args.pipeline

    try:
        logger.info(f"load pipeline from {pipeline_path}")
        pipeline = Pipeline.from_file(pipeline_path)
        pipeline.fit(training_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Pipeline was not found on {pipeline_path}")

    if not pipeline.validate(verbose=True):
        raise RuntimeError("Pipeline is not valid after training")

    model_path = os.path.join(args.model_dir, 'pipeline.pkl')
    logging.info(f"saving to {model_path}")
    pipeline.save(model_path)
    logging.info("training complete")


if __name__ == "__main__":
    main()
