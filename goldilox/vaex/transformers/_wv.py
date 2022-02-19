import logging

import numpy as np
import vaex

logger = logging.getLogger(name='VowpalWabbit')

CATEGORICAL = 'categorical'
TEXT = 'text'


@vaex.register_dataframe_accessor('vw', override=True)
class DataFrameAccessorTensorflow(object):
    def __init__(self, df):
        self.df = df
        self.column_map = None
        self.numeric_features_indices = None
        self.categorical_features_indices = None
        self.text_features_indices = None
        self.target_index = None
        self.weight_index = None
        self.row_length = None

    def init_generator(self, features, target=None, weights=None, categorical_features=[], text_features=[]):
        target = target
        numeric_features_indices = set([])
        categotical_features_indices = set([])
        text_features_indices = set([])
        self.target_index = None
        self.weight_index = None
        non_numeric = categorical_features + text_features
        self.column_map = {i: name for i, name in enumerate(self.df.get_column_names())}
        all_columns = self.df.get_column_names()
        for i, column in enumerate(all_columns):
            if column == target:
                self.target_index = i
            elif column == weights:
                self.weight_index = i
            elif column not in non_numeric and self.df[column].dtype.is_numeric:
                numeric_features_indices.add(i)
            elif column in categorical_features:
                categotical_features_indices.add(i)
            elif column in text_features:
                text_features_indices.add(i)
            elif self.df[column].dtype.is_string:
                categotical_features_indices.add(i)
        assert len(categotical_features_indices) + len(text_features_indices) + len(numeric_features_indices) == len(
            features)
        self.numeric_features_indices = numeric_features_indices
        self.categorical_features_indices = categotical_features_indices
        self.text_features_indices = text_features_indices
        self.row_length = len(features)
        if target is not None:
            self.row_length += 1
        if weights is not None:
            self.row_length += 1

    def to_vw_generator(self, features, target=None, weights=None, categorical_features=[], text_features=[], epochs=5,
                        verbose=True):
        self.init_generator(features=features, target=target, weights=weights,
                            categorical_features=categorical_features, text_features=text_features)

        def _generator(verbose=verbose):
            length = len(self.df)
            for epoch in range(epochs):
                if verbose:
                    logger.info(f"start epoch {epoch}")
                for i in range(length):
                    row = self.df[i]
                    ex = self.to_vw(row)

                    if verbose and i == 0:
                        logger.info(f"example first row {i}: {ex}")
                    yield ex

        return _generator()

    def to_vw(self, row):
        row = list(row)
        weight = ''
        truth = ''
        if len(row) < self.row_length:
            if self.target_index is not None:
                row.insert(self.target_index, truth)
            if self.weight_index is not None:
                row.insert(self.weight_index, weight)
        numeric_strings = []
        categorical_strings = []
        text_strings = []
        for index, value in enumerate(row):
            if value is None:
                continue
            name = self.column_map.get(index)
            if index == self.target_index:
                truth = value
            elif index == self.weight_index:
                weight = value
            elif index in self.numeric_features_indices:
                if np.isnan(value):
                    continue
                numeric_strings.append(f"{name}:{value}")
            elif index in self.categorical_features_indices:
                categorical_strings.append(f"{name}_{str(value).replace(' ', '_')}")
            elif index in self.text_features_indices:
                text_strings.append(f"{value}")
        values = [f"{truth} {weight}"]
        if 0 < len(numeric_strings) > 0:
            values.append(' '.join(numeric_strings))
        if 0 < len(categorical_strings):
            values.append(' '.join([CATEGORICAL] + categorical_strings))
        if 0 < len(text_strings):
            values.append(' '.join([TEXT] + text_strings))

        ex = '|'.join(values)
        return ex
