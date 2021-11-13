from goldilox import Pipeline
from tempfile import TemporaryDirectory

def validate_persistance(pipeline):
    tmpdir = TemporaryDirectory().name
    path = str(tmpdir) + '/model.pkl'
    pipeline.save(path)
    return Pipeline.from_file(path)