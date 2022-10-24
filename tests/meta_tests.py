from goldilox import Meta


def test_dict_conversion():
    meta = Meta('vaex')
    meta = Meta.from_dict(meta.to_dict())
    assert meta.pipeline_type == 'vaex'
    assert meta.py_version == Meta.get_python_version()
