API documentation for vaex library
==================================


Quick lists
-----------

Pipeline.
~~~~~~~~~~~~~~

.. autosummary::

    goldilox.pipeline.Pipeline.from_vaex
    goldilox.pipeline.Pipeline.from_sklearn


CLI.
~~~~~~~~~~~

.. autosummary::

    goldilox.app.cli.serve
    goldilox.app.cli.freeze
    goldilox.app.cli.description
    goldilox.app.cli.example
    goldilox.app.cli.raw
    goldilox.app.cli.build
    goldilox.app.cli.variables
    goldilox.app.cli.packages



.. toctree::

vaex-core
---------

.. automodule:: vaex
    :members: open, concat, from_dict, from_items, from_arrays, from_arrow_table, from_arrow_dataset, from_dataset, from_pandas, from_ascii, from_json, from_records, from_csv, from_astropy_table, open_many, register_function, server, example, app, delayed, vrange, vconstant
    :undoc-members:
    :show-inheritance:


Pipeline class
~~~~~~~~~~~~~~~

.. autoclass:: goldilox.pipeline.Pipeline
     :members:
     :special-members:


SklearnPipeline class
~~~~~~~~~~~~~~~

.. autoclass:: goldilox.sklearn.pipeline.Pipeline
     :members:
     :special-members:

VaexPipeline class
~~~~~~~~~~~~~~~

.. autoclass:: goldilox.vaex.pipeline.Pipeline
     :members:
     :special-members:


Transformers & Encoders
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    goldilox.sklearn.transformers.Imputer


.. autoclass:: goldilox.sklearn.transformers.Imputer
     :members: