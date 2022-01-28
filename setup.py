from setuptools import setup, find_packages

setup(
    name="cord-dataset",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "natsort",
        "numpy>=1.21",
        "torch>=1.10",
        "torchvision",
        "Pillow>=8.4",
        "requests",
        "tqdm",
        "cord-client-python @ git+https://github.com/cord-team/cord-client-python.git@rp/floaty-tqdm"
    ],
    python_requires=">=3.7",
)
