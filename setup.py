from setuptools import setup, find_packages

import goldilox

packages = find_packages()
setup(
    name="goldilox",
    packages=packages,
    include_package_data=True,
    install_requires=["scikit-learn", "pandas", "numpy", "cloudpickle", "traitlets",
                      "gunicorn", "Click", "fastapi", "uvicorn", "pickle5"],
    version=goldilox.__version__,
    url="https://github.com/xdssio/goldilox",
    description="A tool for deploying machine learning",
    author="Yonatan Alexander",
    author_email="jonathan@xdss.io",
    entry_points="""
        [console_scripts]
        glx=goldilox.app.cli:main
    """,
    python_requires='>=3',
)
