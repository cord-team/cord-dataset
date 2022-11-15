# DEPRECATION WARNING

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This repo is currently not actively maintained. New features and formats within the Encord application might break the code here.

Please contact [support@encord.com](mailto:support@encord.com) if you have any questions.
You can also see [the Encord python SDK documentation](https://python.docs.encord.com) to learn more about how to fetch data from the Encord platform.

<hr/>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

***The data engine for computer vision***

This repository holds an implementation of a `Dataset`, which is compatible with the PyTorch framework. The `Dataset`
will fetch information about data, geometries, and labels from the
[`cord-client-python`](https://github.com/cord-team/cord-client-python)
module and structure it into an easily accessible format.

## Installation
```bash
$ python -m pip install git+https://github.com/encord-team/encord-dataset.git
```

### Other requirements
The data loader uses `ffmpeg` to split videos into individual frames.
So make sure that you have `ffmpeg` installed:
```commandline
$ ffmpeg -version
```

## Usage

```python
from encord_dataset import EncordData

config_path = './example_config.ini'
data = EncordData(config_path, download=True)

img, attr = data[0]
print(attr)
```

### Configuration

The `Dataset` depends on a `project_id`, `api_key`, and a `cache_dir`, which can all be specified in
a `<your-config>.ini` file, as done in
`example_config.ini`. 

 
