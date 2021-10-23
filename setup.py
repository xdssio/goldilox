from setuptools import setup

setup(
    name='goldilox',
    packages=['goldilox'],
    install_requires=['sklearn', 'pandas','vaex'],
    version='0.0.1',
    url="git@github.com:xdssio/goldilox.git",
    description='A tool for deploying machine learning',
    author="Yonatan Alexander"
)
