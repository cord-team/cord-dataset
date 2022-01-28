import os
import re
import unicodedata
from configparser import ConfigParser
from pathlib import Path
from typing import Union

import requests
from cord.client import CordClientProject, CordClient
from tqdm import tqdm

from .objects import BoundingBox


def is_frac_bbox(bbox: BoundingBox) -> bool:
    checks = [
        isinstance(bbox.x, float),
        isinstance(bbox.y, float),
        isinstance(bbox.h, float),
        isinstance(bbox.w, float),
        min(bbox) >= 0.0,
        max(bbox) <= 1.0,
    ]
    return all(checks)


def is_pixel_bbox(bbox: BoundingBox, h: int = None, w: int = None) -> bool:
    checks = [
        isinstance(bbox.x, int),
        isinstance(bbox.y, int),
        isinstance(bbox.w, int),
        isinstance(bbox.x, int),
        bbox.x >= 0,
        bbox.y >= 0,
        h is None or bbox.y + bbox.h <= h,
        w is None or bbox.x + bbox.w <= w,
    ]
    return all(checks)


def bbox_frac_to_pixels(bbox: BoundingBox, img_h, img_w) -> BoundingBox:
    if is_pixel_bbox(bbox):
        return bbox

    x = int(bbox.x * img_w)
    y = int(bbox.y * img_h)
    w = int(bbox.w * img_w)
    h = int(bbox.h * img_h)
    return BoundingBox(x=x, y=y, w=w, h=h)


def bbox_pixels_to_frac(bbox, img_h, img_w) -> BoundingBox:
    if is_frac_bbox(bbox):
        return bbox

    x = bbox.x / img_w
    y = bbox.y / img_h
    w = bbox.w / img_w
    h = bbox.h / img_h
    return BoundingBox(x=x, y=y, w=w, h=h)


def download_file(
    url: str,
    destination: Union[str, Path],
    fname: str = None,
    byte_size=1024,
    progress=tqdm,
):
    if not fname:
        fname = url.split("/")[-1]

    local_filename = destination / fname
    iterations = int(int(requests.head(url).headers["Content-Length"]) / byte_size) + 1
    r = requests.get(url, stream=True)

    if progress is None:

        def nop(it, *a, **k):
            return it

        progress = nop

    with open(local_filename, "wb") as f:
        for chunk in progress(
            r.iter_content(chunk_size=byte_size), total=iterations, desc="Downloading "
        ):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
    return local_filename


def slugify(value: str) -> str:
    """Remove all ugly characters from string. (Apply, e.g., for file names."""
    value = str(value)
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_cord_config(config_file: str) -> ConfigParser:
    """
    Args:
        config_file: file name of Cord config file to load.

    Returns: config with project_id, api_key, and cache_dir to use.
    """
    config_file = Path(config_file)
    if not os.path.exists(config_file):
        raise ValueError(f"{config_file.absolute()} does not exist")

    config = ConfigParser()
    config.read(config_file)
    assert (
        "DEFAULT" in config
    ), f"config {config_file} should have section named [DEFAULT]"

    default = config["DEFAULT"]
    assert (
        "project_id" in default
    ), f"config {config_file}:DEFAULT should have key 'project_id'"
    assert (
        "api_key" in default
    ), f"config {config_file}:DEFAULT should have key 'api_key'"
    assert (
        "cache_dir" in default
    ), f"config {config_file}:DEFAULT should have key 'cache_dir'"

    return config


def get_ascii_project_name(project_client: CordClient) -> str:
    """
    Sluggify project title.
    Args:
        project_client: Client to fetch project title from.

    Returns: sluggified project title.
    """
    project = project_client.get_project()
    project_name = slugify(project.get("title"))
    return project_name


def get_cache_dir(config: Union[ConfigParser, str], project_client: CordClient) -> Path:
    """
    Args:
        config: Cord config to extract cache dir from.
        project_client: The projcet client to extract project title from => becomes subdirectory in config cache dir.

    Returns: path to cache dir.
    """
    if isinstance(config, str):
        config = get_cord_config(config)

    cache_dir = config.get("DEFAULT", "cache_dir")
    cache_dir = Path(cache_dir).expanduser().resolve()

    res = cache_dir / get_ascii_project_name(project_client)
    os.makedirs(res, exist_ok=True)

    return res


def get_cord_project_client(config: ConfigParser) -> CordClientProject:
    """
    Initialize Cord client from Cord config.
    Args:
        config: Cord config.

    Returns: project client.
    """
    return CordClient.initialise(
        config["DEFAULT"]["project_id"], config["DEFAULT"]["api_key"]
    )
