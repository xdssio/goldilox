import logging
import os
from contextlib import suppress
from functools import partial
from pathlib import Path

import PIL.Image
import ipyplot
import numpy as np
import tensorflow as tf
import tensorflow_io.arrow as arrow_io
import vaex
from tensorflow.python.keras.models import model_from_json
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import arrow_schema_to_tensor_types
from vaex.ml.state import HasState

logger = logging.getLogger("keras")


class KerasModel(HasState):

    def __init__(self, model=None):
        with suppress():
            model = model_from_json(model)

        self.model = model

    def __reduce__(self):
        return (self.__class__, (self.model.to_json(),))

    def predict(self, ar):
        if isinstance(ar, vaex.expression.Expression):
            ar = ar.values
        return self.model.predict(ar)


@vaex.register_dataframe_accessor('keras', override=True)
class DataFrameAccessorTensorflow(object):
    def __init__(self, df):
        self.df = df

    def _arrow_batch_generator(self, column_names, chunk_size=1024, parallel=True):
        """Create a generator which yields arrow table batches, to use as datasoure for creating Tensorflow datasets.
        :param features: A list of column names.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        Returns:
        :return: generator that yields arrow table batches
        """
        for i1, i2, table in self.df.to_arrow_table(column_names=column_names, chunk_size=chunk_size,
                                                    parallel=parallel):
            yield table.to_batches(chunk_size)[0]

    @staticmethod
    def _get_batch_arrow_schema(arrow_batch):
        """Get the schema from a arrow batch table."""
        output_types, output_shapes = arrow_schema_to_tensor_types(arrow_batch.schema)
        return output_types, output_shapes

    def to_dataset(self, features, target=None, chunk_size=1024, as_dict=True, parallel=True):
        """Create a tensorflow Dataset object from a DataFrame, via Arrow.
        :param features: A list of column names.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param as_dict: If True, the dataset will have the form of dictionary housing the tensors.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        This is useful for making inputs directly for tensorflow. If False, the dataset will contain Tensors,
        useful for passing the dataset as a datasource to a Keras model.
        Returns:
        :return ds: A tensorflow Dataset
        """
        if target is not None:
            target = vaex.utils._ensure_list(target)
            target = vaex.utils._ensure_strings_from_expressions(target)
            n_target_cols = len(target)
            column_names = features + target
        else:
            column_names = features

        # Set up the iterator factory
        iterator_factory = partial(self._arrow_batch_generator, **{'column_names': column_names,
                                                                   'chunk_size': chunk_size,
                                                                   'parallel': parallel})
        # get the arrow schema
        output_types, output_shapes = self._get_batch_arrow_schema(next(iterator_factory()))

        # Define the TF dataset
        ds = arrow_io.ArrowStreamDataset.from_record_batches(record_batch_iter=iterator_factory(),
                                                             output_types=output_types,
                                                             output_shapes=output_shapes,
                                                             batch_mode='auto',
                                                             record_batch_iter_factory=iterator_factory)

        # Reshape the data into the appropriate format
        if as_dict:
            if target is not None:
                if n_target_cols == 1:
                    ds = ds.map(lambda *tensors: (dict(zip(features, tensors[:-1])), tensors[-1]))
                else:
                    ds = ds.map(lambda *tensors: (dict(zip(features, tensors[:-n_target_cols])),
                                                  # dict(zip(target, tensors[-n_target_cols:]))))
                                                  tf.stack(tensors[-n_target_cols:], axis=1)))
            else:
                ds = ds.map(lambda *tensors: (dict(zip(features, tensors))))
        else:
            if target is not None:
                if n_target_cols == 1:
                    ds = ds.map(lambda *tensors: (tf.stack(tensors[:-1], axis=1), tensors[-1]))
                else:
                    ds = ds.map(lambda *tensors: (tf.stack(tensors[:-n_target_cols], axis=1),
                                                  tf.stack(tensors[-n_target_cols:], axis=1)))
            else:
                ds = ds.map(lambda *tensors: (tf.stack(tensors, axis=1)))

        return ds

    def make_input_function(self, features, target=None, chunk_size=1024, repeat=None, shuffle=False, parallel=True):
        """Create a tensorflow Dataset object from a DataFrame, via Arrow.
        :param features: A list of column names.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param repeat: If not None, repeat the dataset as many times as specified.
        :param shuffle: If True, the elements of the dataset are randomly shuffled. If shuffle is True and repeat is not None,
        the dataset will first be repeated, and the entire repeated dataset shuffled.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        Returns:
        :return ds: A tensorflow Dataset
        """
        if repeat is not None:
            assert (isinstance(repeat, int)) & (
                    repeat > 0), 'The "repeat" arg must be a positive integer larger larger than 0.'
            shuffle_buffer_size = chunk_size * repeat
        else:
            shuffle_buffer_size = chunk_size

        def tf_input_function():
            ds = self.to_dataset(features=features, target=target, chunk_size=chunk_size)
            if repeat is not None:
                ds = ds.repeat(repeat)
            if shuffle:
                ds = ds.shuffle(shuffle_buffer_size)

            return ds

        return tf_input_function

    def to_keras_generator(self, features, target=None, chunk_size=1024, parallel=True, verbose=True):
        """Return a batch generator suitable as a Keras datasource.  Note that the generator is infinite, i.e. it loops
        continuously ovder the data. Thus you need to specify the "steps_per_epoch" arg when fitting a keras model,
        the "validation_steps" when using it for validation, and "steps" when calling the "predict" method of a keras model.
        :param features: A list of column names.
        :param target: The dependent or target column or a list of columns, if any.
        :param chunk_size: Number of samples per chunk of data. This can be thought of as the batch size.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        :parallel verbose: If True, show an info on the recommended "steps_per_epoch"
        based on the total number of samples and "chunk_size".
        """
        if verbose:
            steps_per_epoch = np.ceil(len(self.df) / chunk_size)
            logging.info(f'Recommended "steps_per_epoch" arg: {steps_per_epoch}')

        target = target

        def _generator(features, target, chunk_size, parallel):
            if target is not None:
                target = vaex.utils._ensure_list(target)
                target = vaex.utils._ensure_strings_from_expressions(target)
                n_target_cols = len(target)
                column_names = features + target

            else:
                column_names = features
            column_shapes = [self.df[column].shape for column in column_names]
            number_of_features = len(features)

            while True:
                if target is not None:
                    for i1, i2, chunks in self.df.evaluate_iterator(column_names, chunk_size=chunk_size,
                                                                    parallel=parallel):
                        chunks = [np.array(feature_data) for feature_data in chunks]
                        if len(chunks[0]) < chunk_size:
                            return
                        X = chunks[:number_of_features]
                        if len(X) == 1:
                            X = X[0]
                        else:
                            raise NotImplementedError("need to do")
                        y = chunks[-1]
                        yield (X, y)
                else:
                    for i1, i2, chunks, in self.df.evaluate_iterator(column_names, chunk_size=chunk_size,
                                                                     parallel=parallel):
                        X = np.array(chunks).T
                        yield (X,)

        return _generator(features, target, chunk_size, parallel)

    def persists_model_predict(self, model):
        return KerasModel(model).predict

    def show(self, x, labels=None, max_images=20, img_width=85):
        x = vaex.utils._ensure_list(x)[0]
        images = self.df[x].values
        if labels is not None:
            labels = vaex.utils._ensure_list(labels)[0]
            labels = self.df[labels].values
        else:
            labels = [''] * len(images)
        ipyplot.plot_images(images, max_images=max_images, img_width=img_width, labels=labels)
        return None


def open_images(path, suffix=None, resize=None):
    if os.path.isfile(path):
        files = [path]
    if os.path.isdir(path):
        files = []
        if suffix is not None:
            files = [str(path) for path in Path(path).rglob(f"*{suffix}")]
        else:
            for suffix in ('jpg', 'png', 'jpeg', 'ppm', 'thumbnail'):
                files.extend([str(path) for path in Path(path).rglob(f"*{suffix}")])
    num_skipped = 0
    ignores = set([])
    for file in files:
        try:
            fobj = open(file, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            logger.error(f"file {path} is corrupted - ignore")
            ignores.add(file)
    files = [file for file in files if file not in ignores]
    df = vaex.from_arrays(path=files)
    logger.info(f"found {len(files)} files")

    # if shape is None:
    #     shape = np.array(PIL.Image.open(files[0])).shape
    #
    # def read_image(path):
    #     try:
    #         im = PIL.Image.open(path)
    #         im = resize(im, shape)
    #         return np.array(im)
    #     except:
    #         logger.error(f"failed to open - {path}")
    #     return None
    @vaex.register_function()
    def imopen(paths):
        images = [PIL.Image.open(path) for path in vaex.array_types.tolist(paths)]
        return np.array(images, dtype="O")

    df['image'] = df['path'].imopen()

    return df
