from setuptools import setup

setup(
    name='goldilox',
    packages=['goldilox'],
    install_requires=['sklearn', 'pandas', 'vaex', 'numpy'],
    version='0.0.1',
    url="https://github.com/xdssio/goldilox",
    description='A tool for deploying machine learning',
    author="Yonatan Alexander",
    author_email="jonathan@xdss.io"
)
