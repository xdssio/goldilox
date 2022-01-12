from setuptools import setup, find_packages

packages = find_packages()
print(packages)
setup(
    name="goldilox",
    packages=packages,
    include_package_data=True,
    install_requires=["sklearn", "pandas", "numpy", "cloudpickle", "gunicorn", "Click", "fastapi", "uvicorn"],
    version="0.0.4",
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
