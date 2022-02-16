from setuptools import setup, find_packages

setup(
    name="encord-dataset",
    version="0.1.3+fhv.label_filtering",
    packages=find_packages(),
    install_requires=[
        "natsort",
        "numpy>=1.21",
        "torch>=1.10",
        "torchvision",
        "Pillow>=8.4",
        "requests",
        "dacite>=1.6.0",
        "tqdm",
        "cord-client-python @ git+https://github.com/encord-team/encord-client-python.git@rp/floaty-tqdm"
    ],
    python_requires=">=3.7",
)
