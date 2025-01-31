from setuptools import setup, find_packages

setup(
    name='LatentUQ',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'pyyaml',  # or just 'yaml' depending on the package you use
        'scipy'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
)