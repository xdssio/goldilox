import logging
import os

from fastapi import HTTPException


def fix_import():
    import sklearn  # 1
    import lightgbm  # 2
    import catboost  # 3
    assert sklearn.__version__ and lightgbm.__version__ and catboost.__version__


logger = logging.getLogger(__name__)

from goldilox.meta import Meta
from goldilox.app import get_app

path = os.getenv('PIPELINE_PATH')
app = get_app(path)
meta = Meta.from_file(path)


@app.get("/bootstrap", response_model=dict)
def bootstrap():
    logger.info("/bootstrap")
    try:
        ret = {
            'raw': meta.raw.copy(),
            'example': meta.example.copy(),
            'description': meta.description,
            'variables': meta.variables.copy(),
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    return ret


from mangum import Mangum

handler = Mangum(app, lifespan="auto")
