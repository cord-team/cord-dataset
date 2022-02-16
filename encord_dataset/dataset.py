import copy
import json
import logging
import os
import subprocess
import dacite
from enum import Enum
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor as Executor
from configparser import ConfigParser
from pathlib import Path
from typing import Union, Optional, List, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from cord.client import CordClient, CordClientProject
from natsort import natsorted
from torch._utils import _accumulate
from torch.random import Generator
from torch.utils.data import Dataset
from tqdm import tqdm

from .objects import (
    Ontology,
    DataUnit,
    DataUnitObject,
    ClassificationInfo,
    Attributes,
    ImageLabelRow,
    VideoLabelRow,
    SummaryLabelRow,
)
from .transforms import TransformOutput, Transform
from .filters import FilterCollection, FILTER_MODE_INDEX, FilterFactory

# Local imports
from .utils import (
    download_file,
    get_cord_config,
    get_cord_project_client,
    get_cache_dir,
    slugify,
)

logger = logging.getLogger(__name__)

ObjectIndex = namedtuple("ObjectIndex", "ii oi")  # ii: Image index, oi: object index


def check_data_link(data_unit: DataUnit) -> None:
    """Asserts that link can be downloaded."""
    assert (
        data_unit.data_link[:8] == "https://"
    ), f"`data_unit.data_link` not downloadable.\n{data_unit.data_link}"


def extract_frames(video_file_name: Union[str, Path], img_dir: Union[str, Path]):
    logger.info(f"Extracting frames from video: {video_file_name}")
    if isinstance(video_file_name, Path):
        video_file_name = video_file_name.expanduser().resolve().as_posix()
    if isinstance(img_dir, Path):
        img_dir = img_dir.expanduser().resolve().as_posix()

    os.makedirs(img_dir, exist_ok=True)
    command = (
        f"ffmpeg -i {video_file_name} -start_number 0 {img_dir}/%d.jpg -hide_banner"
    )
    if (
        subprocess.run(
            command, shell=True, capture_output=True, stdout=None, check=True
        ).returncode
        != 0
    ):
        logger.warning("Splitting video into separate frames failed.")


def replace_img_with_tensor(img_file: Path, return_tensor=False):
    try:
        if not os.path.exists(img_file.with_suffix(".pt")):
            img = F.to_tensor(ImageOps.exif_transpose(Image.open(img_file)))
            img_ = (img * 255).type(torch.uint8)
            torch.save(img_, img_file.with_suffix(".pt"))
        elif return_tensor:
            return torch.load(img_file.with_suffix(".pt"))

        try:
            img_file.unlink()
        except FileNotFoundError:
            pass

        if return_tensor:
            return img

    except SystemError as e:
        logger.error("Something went wrong when converting images: ", e)
        return False

    return True


def convert_frames_from_jpg_to_tensors(img_dir: Path):
    logger.info(
        "Converting video frames from jpg to pytorch pt files for faster access."
    )
    images = [img_dir / f for f in os.listdir(img_dir) if "jpg" in f]

    with Executor(max_workers=8) as exe:
        jobs = [exe.submit(replace_img_with_tensor, i) for i in images]
        results = [job.result() for job in jobs]

    assert all(results)


def get_data_unit_image(
    data_unit: DataUnit, cache_dir: Path, download: bool = False, force: bool = False
) -> Optional[Path]:
    """
    Fetches image either from cache dir or by downloading and caching image. By default,
    only the image path will be returned as a Path object.
    Args:
        data_unit: The data unit that specifies what image to fetch.
        cache_dir: The directory to fetch cached results from, and to cache results to.
        download: If download is true, download image. Otherwise, return None
        force: Force refresh of cached content.

    Returns: The image as a Path, numpy array, or PIL Image or None if image doesn't
        exist and `download == False`.
    """
    is_video = "video" in data_unit.data_type
    if is_video:
        video_hash, frame_idx = data_unit.data_hash.split("_")
        video_dir = cache_dir / "videos"
        video_file = f"{video_hash}.{data_unit.extension}"
        img_dir = video_dir / video_hash
        img_file = f"{frame_idx}.jpg"

        os.makedirs(video_dir, exist_ok=True)
    else:
        img_dir = cache_dir / "images"
        img_file = f"{data_unit.data_hash}.{data_unit.extension}"

    full_img_pth = img_dir / img_file
    torch_file = (img_dir / img_file).with_suffix(".pt")
    return torch_file

    if not (download or force) and not (full_img_pth.exists() or torch_file.exists()):
        return None

    if force or (download and not (full_img_pth.exists() or torch_file.exists())):
        check_data_link(data_unit)
        if is_video:
            # Extract frames images
            if not os.path.exists(video_dir / video_file):
                logger.info(f"Downloading video {video_file}")
                download_file(
                    data_unit.data_link, video_dir, fname=video_file, progress=None
                )
            extract_frames(video_dir / video_file, img_dir)
            convert_frames_from_jpg_to_tensors(img_dir)
        else:
            logger.debug(f"Downloading image {full_img_pth}")
            download_file(data_unit.data_link, img_dir, fname=img_file, progress=None)
            replace_img_with_tensor(full_img_pth)

    if torch_file.exists():
        return torch_file
    return full_img_pth


def get_label_row(
    label_hash: str,
    project_client: CordClient,
    cache_dir: Path,
    download: bool = False,
    force: bool = False,
) -> dict:
    """
    Fetch label row, either from cache if cache exists (and not `force`) or fetch it
    from the `project_client`.
    Args:
        label_hash: the label hash to fetch.
        project_client: the Cord project client.
        cache_dir: the directory to look for cache files.
        download: only if this is true, data will be downloaded. If false, None will be
        returned for non-cached ids.
        force: if `force`, all label rows will be fetched from scratch
            (with signed urls).

    Returns: Label row in dict format.
    """
    json_file = cache_dir / "labels" / f"{label_hash}.json"

    label_row = None
    if not force and os.path.exists(json_file):
        with open(json_file, "r") as f:
            label_row = json.load(f)

    elif download or force:
        # If force, we assume that signed urls are needed.
        label_row = project_client.get_label_row(label_hash, get_signed_url=True)
        with open(json_file, "w") as f:
            json.dump(label_row, f, indent=2)

    return label_row


def get_data_unit_labels(data_unit: DataUnit) -> List[Attributes]:
    """
    Extract important information from data_unit. That is, get only bounding_boxes and
    associated classifications.
    Args:
        data_unit: The data unit to extract information from.

    Returns: list of pairs of objects and associated answers for the particular data
        unit.
    """
    res = []
    for obj in data_unit.objects:
        # Classifications (are both on the object_answer.classifications and on the object.
        # Store all nested classification info.
        obj_answer = obj.object_answer
        classes = [
            ClassificationInfo(
                ontology_id=obj.ontology_object.id, value=obj.value, name=obj.name
            )
        ]
        queue = obj_answer.classifications
        while len(queue) > 0:
            c = queue.pop(0)
            # Skip text for now.
            if (
                not hasattr(c.ontology_object, "type")
                or c.ontology_object.type == "text"
            ):
                continue

            classes.append(
                ClassificationInfo(
                    ontology_id=c.ontology_object.id, value=c.value, name=c.name
                )
            )
            if (
                c.answers is not None
                and isinstance(c.answers, list)
                and len(c.answers) > 0
            ):
                queue.extend(c.answers)
            elif c.answers is not None:
                raise ValueError(
                    f"I didn't expect to see this. What to do in this situation?\n{c.answers}"
                )

        # Bounding box and polygon
        bbox = obj.bounding_box if hasattr(obj, "bounding_box") else None
        polygon = obj.polygon if hasattr(obj, "polygon") else None
        res.append(Attributes(bbox=bbox, polygon=polygon, classes=classes, du=obj))

    return res


class EncordData(Dataset):
    def __init__(
        self,
        config_file: str,
        download: bool = False,
        force_refresh: bool = False,
        filters: Optional[FilterCollection] = None,
        mode: Optional[str] = None,
        transform=None,
        ignore_object_classes: Optional[List[str]] = None,
    ):
        """
        Cord data complies with the pytorch Dataset and will deliver data in tuples of
        (images, attributes). Data (images and label information) is downloaded and
        cached in the `config_file.DEFAULT.cache_dir` location. In turn, using this
        data-loader may be slow first time it is run, due to downloading data.

        Currently, the data is split into individual objects for each fame, i.e., every
        index of the EncordData corresponds to a particular object in a frame.

        :param config_file: A config file in the format given in the
            `example_config.ini` in the root of this project.
        :param download: If true, data will be downloaded, otherwise, only cached data
            will be used.
        :param force_refresh: To invalidate cached files for force downloading
            everything again, set `force_refresh=True`.
        :param mode: Which samples to select. Can be one of ['reviewed', 'not-reviewed',
            'all']
        :param transform: Transforms to transform the data, similar to
            `torchvision.transform` but potentially also transforming bounding_boxes and
            polygons.
        :param ignore_object_classes: List of names of object classes to ignore (can be
            loaded from config [semicolon separated]).
        """
        self.config: ConfigParser = get_cord_config(config_file)
        self.project_client: CordClientProject = get_cord_project_client(self.config)
        self.cache_dir = get_cache_dir(self.config, self.project_client)
        self.title = slugify(self.project_client.get_project().get("title"))

        self.download: bool = download or force_refresh
        self.force_refresh: bool = force_refresh

        self.transform: Transform = transform

        if filters is None:
            mode = (
                self.config.get("DEFAULT", "mode", fallback="not-reviewed")
                if mode is None
                else mode
            )

            mode = FILTER_MODE_INDEX[mode]
            self.filters = FilterFactory.get_default_filters(mode)

        else:
            self.filters = filters

        self.ignore_object_classes = []
        if ignore_object_classes is not None and len(ignore_object_classes) > 0:
            self.ignore_object_classes = list(
                map(lambda s: s.lower(), ignore_object_classes)
            )

        elif self.config.has_option("DEFAULT", "ignore_object_classes"):
            ignore_str = self.config.get(
                "DEFAULT", "ignore_object_classes", fallback=""
            )
            if ignore_str:
                self.ignore_object_classes = ignore_str.lower().split(";")

        self.ontology: Optional[Ontology] = None
        self.class_to_idx = {}
        self.images: List[Path] = []
        self.objects: List[List[DataUnitObject]] = []
        self.object_idx: List[ObjectIndex] = []

        self.prepare()

    def prepare(self):
        """
        1. (Down)loads label rows and images by either downloading of loading from
            cache.
        2. Prepares images by downloading them if not stored already and extracting
            frames from videos.
        3. (Down)loads object ontologies and labels.
        """
        project = self.project_client.get_project()

        cfg = dacite.Config(cast=[Enum])
        label_rows = [
            dacite.from_dict(data_class=SummaryLabelRow, data=lr, config=cfg)
            for lr in project.get("label_rows")
            if lr is not None
        ]
        label_rows = self.filter_summary_label_rows(label_rows)

        os.makedirs(self.cache_dir / "labels", exist_ok=True)
        os.makedirs(self.cache_dir / "images", exist_ok=True)
        os.makedirs(self.cache_dir / "videos", exist_ok=True)

        def nothing(x, desc):  # void function as replacement of tqdm
            return x

        progress = nothing if not self.download else tqdm

        self.ontology = Ontology(**project.get("editor_ontology"))

        data_units, progress = self.extract_data_units(label_rows, progress)
        self.images: List[Path] = [
            get_data_unit_image(
                du, self.cache_dir, download=self.download, force=self.force_refresh
            )
            for du in progress(data_units, desc="Downloading images")
        ]
        self.objects: List[List[Attributes]] = [
            get_data_unit_labels(du) for du in data_units
        ]

        ignore_classes = self.filter_ontology_objects()

        self.class_to_idx = {o.id: i for i, o in enumerate(self.ontology.objects)}

        self.object_idx = self.construct_object_index(ignore_classes)

    def filter_summary_label_rows(self, label_rows: List[SummaryLabelRow]):
        return [
            lr
            for lr in label_rows
            if all([fn(lr) for fn in self.filters.label_row_filters])
        ]

    def construct_object_index(self, ignore_classes):
        object_idx = []
        for ii, (img, obj) in enumerate(zip(self.images, self.objects)):
            # Skip images that are not downloaded.
            if img is None:
                continue

            for oi, o in enumerate(obj):
                # Skip rows without valid geometry
                if o.bbox is None:
                    continue

                if o.bbox.w == 0 or o.bbox.h == 0:
                    continue

                if o.classes[0].ontology_id in ignore_classes:
                    continue

                # Apply data unit filters for custom filtering.
                if all([fn(o.du) for fn in self.filters.data_unit_filters]):
                    object_idx.append(ObjectIndex(ii, oi))

        return object_idx

    def extract_data_units(self, label_rows, progress):
        # Prepare all data units
        # Filter data units in terms of what is cached
        data_units = []
        logger.info("Preparing data units")
        for lr in progress(label_rows, desc="Downloading labels"):
            full_lr = get_label_row(
                lr.label_hash,
                self.project_client,
                self.cache_dir,
                download=self.download,
                force=self.force_refresh,
            )

            # Skip if label row is empty (download == False and lr not cached)
            if not full_lr:
                continue

            if full_lr.get("data_type").lower() == "video":
                lr = VideoLabelRow(self.ontology, **full_lr)
            else:
                lr = ImageLabelRow(self.ontology, **full_lr)
            data_units.extend(lr.data_units)
        # NB: When new data (labels) are added, this sorting will not be stable.
        data_units = natsorted(data_units, key=lambda x: x.data_hash)
        return data_units, progress

    def filter_ontology_objects(self):
        ignore_classes = set()
        for i in range(
            len(self.ontology.objects) - 1, -1, -1
        ):  # Reverse order for easy remove.
            o = self.ontology.objects[i]
            if (
                o.name.lower() in self.ignore_object_classes
                or o.id in self.ignore_object_classes
            ):
                ignore_classes.add(o.id)
                self.ontology.objects.pop(i)
        return ignore_classes

    @property
    def top_level_object_classes(self):
        return [
            self.class_to_idx[self.objects[i.ii][i.oi].classes[0].ontology_id]
            for i in self.object_idx
        ]

    @property
    def num_top_level_classes(self):
        return len(self.ontology.classes)

    @property
    def object_class_distribution(self):
        """
        Returns: (list of object ids, list of frequencies)
        """
        present_ids, counts = np.unique(
            self.top_level_object_classes, return_counts=True
        )

        all_counts = [0] * self.num_top_level_objects
        all_ids = list(range(self.num_top_level_objects))

        for i, id in enumerate(all_ids):
            if id in present_ids:
                all_counts[i] = counts[present_ids == id].item()

        return all_ids, all_counts

    @property
    def num_top_level_objects(self):
        return len(self.ontology.objects)

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_objects(self):
        return len(self)

    def __len__(self):
        return len(self.object_idx)

    def __getitem__(self, idx) -> TransformOutput:
        oidx = self.object_idx[idx]

        img_pth: Path = self.images[oidx.ii]
        if img_pth.suffix == ".jpg":
            img = replace_img_with_tensor(img_pth, return_tensor=True)
            self.images[oidx.ii] = img_pth.with_suffix(".pt")
        else:
            img = torch.load(img_pth).float() / 255.0

        attr = self.objects[oidx.ii][oidx.oi]
        attr = copy.deepcopy(attr)

        output = TransformOutput(img=img, attributes=attr)
        if self.transform:
            output = self.transform(output.img, output.attributes)

        return output


# Legacy usage.
CordData = EncordData


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: List[EncordData]):
        super().__init__(datasets)
        self.validate()
        self.title = ";".join([d.title for d in self.datasets])

    def validate(self):
        """
        Will compare ontologies to check that they match.
        """
        ontologies = [d.ontology for d in self.datasets]
        o_ = ontologies.pop(0)
        for o in ontologies:
            assert len(o_.classes) == len(
                o.classes
            ), "Not the same number of classes in the ontologies."
            assert len(o_.objects) == len(
                o.objects
            ), "Not the same number of objects in the ontologies."
            for i in range(len(o_.objects)):
                attributes = ["id", "name", "shape"]
                for attr in attributes:
                    att1 = o_.objects[i].__getattribute__(attr)
                    att2 = o.objects[i].__getattribute__(attr)
                    assert (
                        att1 == att2
                    ), f"Attribute {attr} does not match between ontologies: {att1} vs. {att2}"

    def __getattr__(self, attribute_name):
        """
        This is a bit hacky. If the attribute is not in this object. Go look in one of the sub datsets.
        This gives access to, e.g., ontology and the like.
        """
        try:
            getattr(super(), attribute_name)
        except AttributeError:
            return getattr(self.datasets[0], attribute_name)

    @property
    def top_level_object_classes(self):
        """
        Will return the object class at the top level of the object ontology.
        """
        dataset_classes = [d.top_level_object_classes for d in self.datasets]
        all_classes = []
        for cls in dataset_classes:
            all_classes.extend(cls)
        return all_classes

    @property
    def object_class_distribution(self):
        """
        Returns: (list of object ids, list of frequencies)
        """
        ids = None
        count_list = []
        for ds in self.datasets:
            ids, counts = ds.object_class_distribution
            count_list.append(counts)

        return ids, np.sum(count_list, 0)

    @property
    def num_images(self):
        """
        Describe the number of images/frames available in the dataset.
        """
        return sum([ds.num_images for ds in self.datasets])

    @property
    def num_objects(self):
        """
        Describe the number of objects available in the dataset.
        """
        return sum([ds.num_objects for ds in self.datasets])


class Subset(torch.utils.data.Subset):
    """
    Subset of EncordData or ConcatData to use only some indices, e.g., for train/val split.
    """

    def __init__(self, dataset: EncordData, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)

    @property
    def top_level_object_classes(self):
        ds_classes = self.dataset.top_level_object_classes
        return [ds_classes[i] for i in self.indices]

    @property
    def object_class_distribution(self):
        """
        Returns: (list of object ids, list of frequencies)
        """
        present_ids, counts = np.unique(
            self.top_level_object_classes, return_counts=True
        )

        all_counts = [0] * self.num_top_level_objects
        all_ids = list(range(self.num_top_level_objects))

        for i, id in enumerate(all_ids):
            if id in present_ids:
                all_counts[i] = counts[present_ids == id].item()

        return all_ids, all_counts

    @property
    def num_images(self):
        object_idx = map(lambda x: self.dataset.object_idx[x], self.indices)
        return len(set([self.dataset.images[i.ii] for i in object_idx]))

    @property
    def num_top_level_objects(self):
        return len(self.dataset.ontology.objects)

    @property
    def num_objects(self):
        return len(self)

    def __getattr__(self, attribute_name):
        try:
            return getattr(super(), attribute_name)
        except AttributeError:
            return getattr(self.dataset, attribute_name)


def random_split(
    dataset: EncordData, lengths: Sequence[int], generator: Generator
) -> List[Subset]:
    r"""
    FHV: Modification of torch impolementation.
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]
