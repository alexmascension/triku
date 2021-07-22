import sys

if sys.version_info < (3,):
    sys.exit("scanpy requires Python >= 3.6")

from setuptools import setup, find_packages

try:
    from triku import __author__, __email__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = __version__ = ""

setup(
    name="triku_nb_code",
    version=__version__,
    description="Code to run triku notebooks.",
    long_description="",
    url="https://gitlab.com/alexmascension/triku",
    author=__author__,
    author_email=__email__,
    license="BSD",
    python_requires=">=3.6",
    install_requires=[],
    packages=find_packages(),
    # `package_data` does NOT work for source distributions!!!
    # you also need MANIFTEST.in
    # https://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute
    # package_data={'': '*.txt'},
    # include_package_data=True,
    # entry_points=dict(
    #     console_scripts=['triku=triku.cli.cli_triku:main', 'triku-plotentropy=triku.cli.cli_plot_entropy:main'],
    # ),
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
