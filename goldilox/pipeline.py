from copy import deepcopy
from hashlib import sha256


class Pipeline:

    pipeline_type: str
    example: dict

    @classmethod
    def check_hash(cls, file_path):
        h = sha256()

        with open(file_path, 'rb') as file:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

    def from_vaex(self, df, example=None, fit=None, warnings=True):
        from goldilox.vaex.pipeline import Pipeline as VaexPipeline
        return VaexPipeline.from_dataframe(df=df, example=example, fit=fit, warnings=warnings)

    def from_sklearn(self, df, fit=None, warnings=True):
        from goldilox.sklearn.pipeline import Pipeline as VaexPipeline
        return VaexPipeline.from_dataframe(df=df, fit=fit, warnings=warnings)

    @classmethod
    def _from_koalas(cls, df, **kwargs):
        # from goldilocks.koalas.pipeline import Pipeline as KoalasPipeline
        return deepcopy(df.pipeline)

    def fit(self, df, **kwargs):
        pass

    def transform(self, df, **kwargs):
        pass

    def predict(self, df, **kwargs):
        pass

    def inference(self, df, **kwargs):
        pass

    def infer(self, df):
        raise RuntimeError(f"Not implemented for {self.pipeline_type}")