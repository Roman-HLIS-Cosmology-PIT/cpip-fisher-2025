from setuptools import setup, find_packages

setup(
        name="templatecode",
        version="0.0",
        packages=find_packages(),
        install_requires=[
                'numpy',
                'argparse'
        ]
)  