import sys

if sys.version_info < (3,):
    sys.exit("triku requires Python >= 3.7")
from pathlib import Path

from setuptools import setup, find_packages


__author__ = ", ".join(["Alex M. Ascensión"])
__email__ = ", ".join(
    [
        "alexmascension@gmail.com",
        # We don’t need all, the main authors are sufficient.
    ]
)
__version__ = "2.0.1"


setup(
    name="triku",
    version=__version__,
    description="Feature selection method for Single Cell data.",
    long_description=Path("README.md").read_text("utf-8"),
    url="https://gitlab.com/alexmascension/triku",
    author=__author__,
    author_email=__email__,
    license="BSD",
    python_requires=">=3.7",
    install_requires=[
        module.strip()
        for module in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    packages=find_packages(),
    # `package_data` does NOT work for source distributions!!!
    # you also need MANIFTEST.in
    # https://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute
    # package_data={'': '*.txt'},
    # include_package_data=True,
    entry_points=dict(),
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
