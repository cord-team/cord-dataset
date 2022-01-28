<h1 align="center">
  <p align="center">Cord Dataset for PyTorch</p>
  <a href="https://cord.tech"><img src="https://app.cord.tech/CordLogo.svg" width="150" alt="Cord logo"/></a>
</h1>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

***Where the world creates and manages training data***

This repository holds an implementation of a `Dataset`, which is compatible with the PyTorch framework. The `Dataset`
will fetch information about data, geometries, and labels from the
[`cord-client-python`](https://github.com/cord-team/cord-client-python)
module and structure it into an easily accessible format.

## Installation
```bash
$ python -m pip install git+https://github.com/cord-team/cord-dataset.git
```

### Other requirements
The data loader uses `ffmpeg` to split videos into individual frames.
So make sure that you have `ffmpeg` installed:
```commandline
$ ffmpeg -version
```

## Usage

```python
from cord_dataset import CordData

config_path = './example_config.ini'
data = CordData(config_path, download=True)

img, attr = data[0]
print(attr)
```

### Configuration

The `Dataset` depends on a `project_id`, `api_key`, and a `cache_dir`, which can all be specified in
a `<your-config>.ini` file, as done in
`example_config.ini`. 

 