# import pickle
# from base64 import b64encode
#
# registry = {}
#
#
# def register(cls):
#     registry[fullname(cls)] = cls
#     return cls
#
#
# def fullname(cls):
#     return cls.__module__ + '.' + cls.__name__
#
#
# def to_dict(obj):
#     return dict(cls=fullname(obj.__class__), state=obj.state_get())
#
#
# def from_dict(d, trusted=False):
#     cls_name = d['cls']
#     if cls_name not in registry:
#         # lets load the module, so we give it a chance to register
#         module, dot, cls = cls_name.rpartition('.')
#         __import__(module)
#     if cls_name not in registry:
#         raise ValueError('unknown class: ' + cls_name)
#     else:
#         obj = registry[cls_name].state_from(d['state'], trusted=trusted)
#         # obj.state_set(d['state'])
#         return obj
#
#
# def f(test=None):
#     return 1
#
#
# def encode_function(f):
#     return b64encode(pickle.dumps(f)).decode('ascii')
#
#
# def decode_function(encoding):
#     # blob = encoding.get_blob(spec['blob'])
#
#     return pickle.loads(blob)
#
#     return encoding.decode('pickle', spec['func'], trusted=trusted),
#
#
# encoding = b64encode(pickle.dumps(f)).decode('ascii')
