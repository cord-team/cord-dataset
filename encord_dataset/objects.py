from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Any, Union, Optional

import numpy as np


class BoundingBox(object):
    def __init__(self, x, y, w, h):
        t = type(x)
        object.__setattr__(self, "x", max(t(0), x))
        object.__setattr__(self, "y", max(t(0), y))
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "h", h)

    def __setattr__(self, key, value):
        raise NotImplementedError

    def __delattr__(self, item):
        raise NotImplementedError

    def __iter__(self):
        return iter([self.x, self.y, self.w, self.h])

    def __str__(self):
        return f"BoundingBox(x={self.x} y={self.y} w={self.w} h={self.h})"

    def __repr__(self):
        return f"BoundingBox(x={self.x} y={self.y} w={self.w} h={self.h})"


class OntologyAttribute:
    def __init__(
        self,
        parent: Union[OntologyObject, OntologyAttribute],
        *,
        id: str,
        featureNodeHash: str,
        type: Optional[str] = None,
        required: Optional[bool] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        options: Optional[list] = None,
        value: Optional[str] = None,
        out_map: dict,
        **kwargs,
    ):

        self.id = id
        self.name = name if name else label
        self.type = type
        self.required = required
        self.feature_hash = featureNodeHash
        self.value = value
        self.parent = parent
        self.options = options
        self.rest = kwargs

        if options:
            self.options = [
                OntologyAttribute(self, **o, out_map=out_map) for o in options
            ]

        if out_map is not None:
            out_map[featureNodeHash] = self


class OntologyObject:
    def __init__(
        self,
        parent: Ontology,
        *,
        id: str,
        name: str,
        shape: str,
        featureNodeHash: str,
        attributes: Optional[List] = None,
        out_map: dict,
        **kwargs,
    ):

        self.id = id
        self.name = name
        self.shape = shape
        self.feature_hash = featureNodeHash
        self.parent = parent
        self.attributes = attributes
        self.rest = kwargs
        if self.attributes:
            self.attributes = [
                OntologyAttribute(self, **a, out_map=out_map) for a in attributes
            ]

        if out_map is not None:
            out_map[featureNodeHash] = self


class OntologyOption:
    def __init__(
        self,
        parent: OntologyClass,
        *,
        id: str,
        label: str,
        value: str,
        featureNodeHash: str,
        out_map,
        **kwargs,
    ):
        self.parent = parent
        self.id = id
        self.label = label
        self.value = value
        self.rest = kwargs

        if out_map is not None:
            out_map[featureNodeHash] = self


class OntologyClass:
    def __init__(
        self,
        parent: Union[Ontology, OntologyClass],
        *,
        id: str,
        featureNodeHash: str,
        name: Optional[str] = None,
        options: Optional[list] = None,
        attributes: Optional[list] = None,
        type: Optional[str] = None,
        required: Optional[bool] = None,
        out_map,
        **kwargs,
    ):

        self.required = required
        self.type = type
        self.id = id
        self.name = name
        self.feature_hash = featureNodeHash
        self.parent = parent
        self.subclasses = None
        self.rest = kwargs
        if options:
            self.options = [OntologyOption(self, **v, out_map=out_map) for v in options]

        if attributes:
            self.subclasses = [
                OntologyClass(self, **v, out_map=out_map) for v in attributes
            ]

        if out_map is not None:
            out_map[featureNodeHash] = self


class Ontology:
    def __init__(self, objects: list, classifications: list, **kwargs):
        # self.objects = {k: OntologyObject(*v) for k, v in objects.items()}
        self.out_map = {}
        self.rest = kwargs

        self.objects = []
        for obj in objects:
            self.objects.append(OntologyObject(self, **obj, out_map=self.out_map))

        self.classes = []
        for cls in classifications:
            self.classes.append(OntologyClass(self, **cls, out_map=self.out_map))

    def get_entry(
        self, feature_hash: str
    ) -> Union[OntologyAttribute, OntologyClass, OntologyObject]:
        return self.out_map.get(feature_hash)


class ClassificationAnswer:
    def __init__(
        self,
        ontology: Ontology,
        *,
        name: str,
        value: str,
        featureHash: str,
        manualAnnotation: bool,
        answers: Optional[list] = None,
        **kwargs,
    ):
        self.name = name
        self.value = value
        self.ontology_object = ontology.get_entry(featureHash)
        self.answers = answers
        self.rest = kwargs

        if answers is not None and isinstance(answers, list):
            self.answers = [
                ClassificationAnswer(ontology, manualAnnotation=manualAnnotation, **c)
                for c in answers
            ]

    def has_answers(self) -> bool:
        return self.answers is not None


class ObjectAnswer:
    def __init__(
        self,
        ontology: Ontology,
        *,
        objectHash: str,
        classifications: list,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.object_hash = objectHash
        self.classifications = [
            ClassificationAnswer(ontology, **v) for v in classifications
        ]
        self.name = name
        self.rest = kwargs

    def has_answers(self) -> bool:
        return self.classifications is not None and len(self.classifications) > 0


class ClassificationAnswers:
    def __init__(
        self,
        ontology: Ontology,
        *,
        classificationHash: str,
        classifications: list,
        **kwargs,
    ):
        self.classificationHash = classificationHash
        self.classifications = [
            ClassificationAnswer(ontology, **v) for v in classifications
        ]
        self.rest = kwargs


class DataUnitObject:
    def __init__(
        self,
        ontology: Ontology,
        label_row: LabelRow,
        parent: DataUnit,
        *,
        objectHash: str,
        featureHash: str,
        name: str,
        value: str,
        shape: str,
        manualAnnotation: bool,
        reviews: list,
        confidence: Optional[float] = None,
        **kwargs,
    ):
        self.parent = parent
        self.object_answer = label_row.object_answers[objectHash]
        self.ontology_object = ontology.get_entry(featureHash)
        self.name = name
        self.value = value
        self.shape = shape
        self.manual_annotation = manualAnnotation
        self.reviews = reviews
        self.confidence = confidence
        self.rest = kwargs
        self.is_reviewed = False

        if self.reviews:
            self.is_reviewed = True
            self.is_correct = reviews[-1].get("approved")

        if shape == "bounding_box":
            bbox = kwargs.pop("boundingBox")
            if bbox["x"] is not None:
                self.bounding_box = BoundingBox(**bbox)

        if shape == "polygon":
            p = kwargs.pop("polygon")
            self.polygon = np.array(
                [[p[str(i)]["x"], p[str(i)]["y"]] for i in range(len(p))]
            ).clip(min=0.0)
        if shape == "point":
            self.point = kwargs.pop("point")


class DataUnitClassification:
    def __init__(
        self,
        ontology: Ontology,
        label_row: LabelRow,
        *,
        name: str,
        confidence: float,
        featureHash: str,
        classificationHash: str,
        manualAnnotation: str,
        reviews: list,
        value: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self.value = value
        self.confidence = (confidence,)
        self.ontology_class = ontology.get_entry(featureHash)
        self.classification_answer = label_row.classification_answers[
            classificationHash
        ]
        self.manualAnnotation = manualAnnotation
        self.reviews = reviews
        self.rest = kwargs

        if self.reviews:
            self.is_reviewed = True
            self.is_correct = reviews[-1].get("approved")


class DataUnit:
    def __init__(
        self,
        ontology: Ontology,
        label_row: LabelRow,
        objects: list,
        classifications: list,
        *,
        data_hash: str,
        data_type: str,
        data_link: Optional[str] = None,
        **kwargs,
    ):
        self.data_hash = data_hash
        self.data_link = data_link
        self.data_type = data_type
        self.label_row = label_row
        self.extension = f"{data_type.lower().split('/')[-1]}"
        self.rest = kwargs

        self.objects = [DataUnitObject(ontology, label_row, self, **o) for o in objects]
        self.classifications = [
            DataUnitClassification(ontology, label_row, **c) for c in classifications
        ]


class LabelRow(object):
    def __init__(
        self,
        ontology: Ontology,
        *,
        label_hash: str,
        object_answers: dict,
        classification_answers: dict,
        **kwargs,
    ):
        self.label_hash = label_hash
        self.object_answers = {
            k: ObjectAnswer(ontology, **v) for k, v in object_answers.items()
        }
        self.classification_answers = {
            k: ClassificationAnswers(ontology, **v)
            for k, v in classification_answers.items()
        }
        self.rest = kwargs


class ImageLabelRow(LabelRow):
    def __init__(self, ontology: Ontology, *, data_units: dict, **kwargs):
        super().__init__(ontology, **kwargs)
        self.data_units: List[Optional[DataUnit]] = [None] * len(data_units)
        for i, du in enumerate(data_units.values()):
            labels = du.pop("labels")
            objects, classifications = labels.pop("objects"), labels.pop(
                "classifications"
            )
            self.data_units[i] = DataUnit(
                ontology, self, objects, classifications, **du
            )


class VideoLabelRow(LabelRow):
    def __init__(self, ontology: Ontology, *, data_units: dict, **kwargs):
        super().__init__(ontology, **kwargs)

        assert len(data_units) == 1
        data_unit = data_units[list(data_units.keys())[0]]
        data_hash = data_unit.pop("data_hash")

        # Construct a data unit for every frame
        labels = data_unit.pop("labels")
        self.data_units = [None] * len(labels)
        for i, (k, label) in enumerate(labels.items()):
            objects = label.get("objects")
            classifications = label.get("classifications")
            du = DataUnit(
                ontology,
                self,
                objects,
                classifications,
                data_hash=f"{data_hash}_{k}",
                **data_unit,
            )
            self.data_units[i] = du


# === Summary label rows === #
class FileType(Enum):
    IMAGE_GROUP = "IMG_GROUP"
    VIDEO = "VIDEO"


@dataclass
class SummaryLabelRow:
    data_hash: str
    data_title: str
    data_type: FileType
    dataset_hash: str
    label_hash: Optional[str]
    label_status: str


# # # # # DATACLASS AND TRANSFORM OBJECTS # # # # # # #
@dataclass
class ClassificationInfo:
    ontology_id: str = None
    name: str = None
    value: str = None


@dataclass
class Attributes:
    bbox: BoundingBox = None
    polygon: np.ndarray = None
    classes: List[Any] = None
    du: DataUnitObject = None
