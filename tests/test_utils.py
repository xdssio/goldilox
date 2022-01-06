from tempfile import TemporaryDirectory

from goldilox import Pipeline


def validate_persistence(pipeline):
    return Pipeline.from_file(pipeline.save(str(TemporaryDirectory().name) + '/model.pkl'))
