from setuptools import setup, find_packages

setup(
    name="cord-dataset",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "natsort",
        "numpy>=1.21",
        "torch>=1.10",
        "torchvision",
        "Pillow>=8.4",
        "requests",
        "tqdm",
        "cord-client-python @ git+https://github.com/encord-team/encord-client-python.git@rp/floaty-tqdm"
    ],
    python_requires=">=3.7",
)
