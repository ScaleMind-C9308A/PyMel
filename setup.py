from setuptools import setup, find_packages

from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="pymel",
    version="0.0.0",
    description="",
    long_description=long_description,
    long_description_content_type="",
    url="",
    author="Khoi Do",
    author_email="khoido8899@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["pymel"],
    include_package_data=True,
    install_requires=["numpy", "tqdm", "pyarrow", "fastparquet"]
)