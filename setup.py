from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="DHEERAJ",
    description="A package for dvc ml pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dk6304273089/DHEE",
    author_email="dk6304273089@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=['dvc','pandas','scikit-learn']
)