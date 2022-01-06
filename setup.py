from setuptools import setup, find_packages

setup(
    name="goldilox",
    packages=find_packages(),
    install_requires=["sklearn", "pandas", "numpy", "cloudpickle", "gunicorn"],
    version="0.0.1a6",
    url="https://github.com/xdssio/goldilox",
    description="A tool for deploying machine learning",
    author="Yonatan Alexander",
    author_email="jonathan@xdss.io",
    entry_points="""
        [console_scripts]
        glx=goldilox.app.cli:main
    """,
)
