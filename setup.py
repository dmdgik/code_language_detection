from template import project_name
from setuptools import setup, find_packages

setup(
    name=project_name,
    version='0.1',
    package_dir={"" : "src"},
    packages=find_packages(where="src"),
)
