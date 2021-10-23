import uuid
from collections import OrderedDict

import numpy as np
import six
from vaex.expression import Expression


def is_ndarray(value):
    return isinstance(value, np.ndarray)


def is_numpy_numeric(dtype):
    if is_expression(dtype):
        dtype = dtype.dtype
    return is_numpy_float(dtype) or is_numpy_int(dtype)


def is_numpy_datetime(dtype):
    if is_expression(dtype):
        dtype = dtype.dtype
    return hasattr(dtype, 'char') and dtype.char in np.typecodes.get('Datetime')


def is_numpy_int(dtype):
    if is_expression(dtype):
        dtype = dtype.dtype
    return hasattr(dtype, 'char') and dtype.char in np.typecodes.get('AllInteger')


def is_numpy_float(dtype):
    if is_expression(dtype):
        dtype = dtype.dtype
    return hasattr(dtype, 'char') and dtype.char in np.typecodes.get('AllFloat')


def is_hidden(s):
    return str(s).startswith('__')


def is_expression(e):
    return isinstance(e, Expression)


def unqiue_with_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _ensure_string_from_expression(expression):
    import vaex.expression
    if expression is None:
        return None
    elif isinstance(expression, bool):
        return expression
    elif isinstance(expression, six.string_types):
        return expression
    elif isinstance(expression, vaex.expression.Expression):
        return expression.expression
    else:
        raise ValueError('%r is not of string or Expression type, but %r' % (expression, type(expression)))


def validate_list(obj):
    if obj is not None and not isinstance(obj, list):
        return [obj]
    return obj


class OrderedIndexedDict(OrderedDict):

    def get(self, item, defualt=None):
        ret = super(OrderedIndexedDict, self).get(item, defualt)
        if ret is None and isinstance(item, int):
            keys = list(self.keys())
            ret = super(OrderedIndexedDict, self).get(keys[item], defualt)
        return ret

    def __getitem__(self, item):
        return self.get(item)


def get_random_id():
    return str(uuid.uuid4()).split('-')[0]


def is_float(value):
    try:
        return float(value)
    except ValueError:
        return False
