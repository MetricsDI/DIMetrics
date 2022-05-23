import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().strip().splitlines()
    required = [i for i in required if i[0] != '-']


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dime",
    version="0.0.1",
    author="Swapnil Gupta, Jonathan DeGange, Zhuoyu Han, Adam Karwan, Krzysztof Wilkosz",
    author_emails={"500swapnil@gmail.com", "jdegange85@gmail.com", "krzysztof.wilkosz@gmail.com"},
    description="A library for computing Document Intelligence metrics for key-value pair extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    package_data={"dime": [], "dime.hm": []},
    packages=[
        "dime", "dime.hm"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    python_requires=">=3.6",
)
