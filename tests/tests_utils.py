from tempfile import TemporaryDirectory

import goldilox
from goldilox import Pipeline


def validate_persistence(pipeline):
    return Pipeline.from_file(pipeline.save(str(TemporaryDirectory().name) + '/model.pkl'))


def test_list_files():
    assert len(goldilox.utils.list_files('data', 'titanic')) == 3


def test_read_sklearn_data():
    data = goldilox.utils.read_sklearn_data('data', prefix='titanic', suffix='csv')
    assert data.shape == (1782, 12)


def test_read_sklearn_data():
    data = goldilox.utils.read_vaex_data('data', prefix='titanic', suffix='csv')
    assert data.shape == (1782, 12)

    data = goldilox.utils.read_vaex_data('data', prefix='titanic')
    assert data.shape == (2673, 13)  # plus index for some reason
