import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union, Optional

from .objects import SummaryLabelRow, OntologyObject, DataUnitObject

logger = logging.getLogger(__name__)


# === Abstract methods === #
class Filter(ABC):
    @abstractmethod
    def include(self, obj: Any) -> bool:
        pass

    def __call__(self, obj: Any):
        return self.include(obj)


class SummaryLabelRowFilter(Filter):
    @abstractmethod
    def include(self, obj: SummaryLabelRow):
        pass


class OntologyObjectFilter(Filter):
    @abstractmethod
    def include(self, obj: OntologyObject):
        pass


class DataUnitObjectFilter(Filter):
    @abstractmethod
    def include(self, obj: DataUnitObject):
        pass


class LabelRowStatusFilter(SummaryLabelRowFilter):
    """
    Includes all label rows which have `label_status` in label_status.
    """

    def __init__(
        self,
        legal_status: Optional[Union[str, list, set]] = None,
        illegal_status: Optional[Union[str, list]] = None,
    ):
        if legal_status is None and illegal_status is None:
            raise ValueError(
                "Either `legal_status` OR `illegal_status` should be specified."
            )
        elif legal_status is not None and illegal_status is not None:
            raise ValueError(
                "`legal_status` and `illegal_status` cannot both be defined at once."
            )

        self.legal_objects = legal_status is not None

        status_list = legal_status if self.legal_objects else illegal_status
        if isinstance(status_list, str):
            status_list = status_list.split(";")

        self.status_list = set(status_list)

    def include(self, obj: SummaryLabelRow):
        res = obj.label_status in self.status_list

        if not self.legal_objects:
            res = not res

        logger.debug(
            f"Returning {res} for label row [{obj.label_hash[:8]}...] "
            f"with status {obj.label_status}"
        )
        return res


class DataUnitObjectReviewedFilter(DataUnitObjectFilter):
    def __init__(self, select_reviewed=True):
        self.select_reviewed = select_reviewed

    def include(self, obj: DataUnitObject):
        if self.select_reviewed:
            return obj.is_reviewed and obj.is_correct
        else:
            return not obj.is_reviewed


class IncludeAllFilter(Filter):
    def include(self, obj: Any):
        return True


class FilterMode(Enum):
    ALL = "all"
    NOT_REVIEWED = "not-reviewed"
    REVIEWED = "reviewed"


FILTER_MODE_INDEX = {e.value: e for e in FilterMode}


@dataclass
class FilterCollection:
    label_row_filters: List[Union[SummaryLabelRowFilter, IncludeAllFilter]]
    ontology_filters: List[Union[OntologyObjectFilter, IncludeAllFilter]]
    data_unit_filters: List[Union[DataUnitObjectFilter, IncludeAllFilter]]


class FilterFactory:
    @staticmethod
    def get_default_filters(mode: FilterMode) -> FilterCollection:
        label_row_filters = [LabelRowStatusFilter(illegal_status="LABEL_IN_PROGRESS")]
        ontology_filters = []

        if mode == FilterMode.ALL:
            data_unit_filters = [IncludeAllFilter()]
        elif mode == FilterMode.REVIEWED:
            data_unit_filters = [DataUnitObjectReviewedFilter(select_reviewed=True)]
        elif mode == FilterMode.NOT_REVIEWED:
            data_unit_filters = [DataUnitObjectReviewedFilter(select_reviewed=True)]
        else:
            raise ValueError("Wrong filter mode")

        return FilterCollection(
            label_row_filters,
            ontology_filters,
            data_unit_filters,
        )
