import logging

from fastapi import HTTPException


def fix_import():
    import sklearn  # 1
    import lightgbm  # 2
    import catboost  # 3
    assert sklearn.__version__ and lightgbm.__version__ and catboost.__version__


logger = logging.getLogger(__name__)

from goldilox.meta import Meta
from goldilox.app import get_app

app = get_app('pipeline.pkl')
meta = Meta.from_file('/opt/program/pipeline.pkl')


@app.get("/bootstrap", response_model=dict)
def bootstrap():
    logger.info("/bootstrap")
    try:

        ret = {
            'raw': meta.raw.copy(),
            'example': meta.example.copy(),
            'description': meta.description
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    return ret


from mangum import Mangum

handler = Mangum(app, lifespan="auto")
