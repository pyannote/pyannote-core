from abc import ABCMeta, abstractmethod
from typing import Optional, Iterator, Tuple, Union, Dict, TYPE_CHECKING, Callable, List

from sortedcontainers import SortedList
from typing_extensions import Self

from pyannote.core import Segment
from pyannote.core.utils.types import Support, CropMode, ContiguousSupport

if TYPE_CHECKING:
    from .timeline import Timeline


class BaseSegmentation(metaclass=ABCMeta):
    """Abstract base  class for all segmented annotations"""

    def __init__(self, uri: Optional[str] = None):
        # path to (or any identifier of) segmented resource
        self._uri: Optional[str] = uri

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, uri: str):
        self._uri = uri

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __nonzero__(self):
        pass

    @abstractmethod
    def __bool__(self):
        """Truthiness of the segmentation. Truthy means that it contains something
        False means it's empty."""
        pass

    @abstractmethod
    def __eq__(self, other: Self):
        pass

    @abstractmethod
    def __ne__(self, other: Self):
        pass

    @abstractmethod
    def itersegments(self):
        pass

    def get_timeline(self) -> 'Timeline':
        from .timeline import Timeline
        return Timeline(self.itersegments())

    @abstractmethod
    def update(self, other: Self) -> Self:
        pass

    def co_iter(self, other: Union['BaseSegmentation', Segment]) -> Iterator[Tuple[Segment, Segment]]:
        if isinstance(other, Segment):
            other_segments = SortedList([other])
        else:
            other_segments = SortedList(other.itersegments())

        for segment in self.itersegments():

            # iterate over segments that starts before 'segment' ends
            temp = Segment(start=segment.end, end=segment.end)
            for other_segment in other_segments.irange(maximum=temp):
                if segment.intersects(other_segment):
                    yield segment, other_segment

    @abstractmethod
    def get_overlap(self) -> 'Timeline':
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __contains__(self, included: Union[Segment, 'Timeline']) -> bool:
        pass

    @abstractmethod
    def empty(self) -> Self:
        pass

    @abstractmethod
    def copy(self, segment_func: Optional[Callable[[Segment], Segment]] = None) \
            -> Self:
        pass

    @abstractmethod
    def extent(self) -> Segment:
        pass

    @abstractmethod
    def crop_iter(self,
                  support: Support,
                  mode: CropMode = 'intersection',
                  returns_mapping: bool = False) \
            -> Iterator[Union[Tuple[Segment, Segment], Segment]]:
        pass

    @abstractmethod
    def duration(self) -> float:
        pass

    @abstractmethod
    def _repr_png_(self):
        pass


class GappedAnnotationMixin(metaclass=ABCMeta):

    @abstractmethod
    def gaps_iter(self, support: Optional[Support] = None) -> Iterator[Segment]:
        pass

    @abstractmethod
    def gaps(self, support: Optional[Support] = None) -> 'Timeline':
        pass

    @abstractmethod
    def extrude(self,
                removed: Support,
                mode: CropMode = 'intersection') -> Self:
        pass


    @abstractmethod
    def crop(self,
             support: Support,
             mode: CropMode = 'intersection',
             returns_mapping: bool = False) \
            -> Union[Self, Tuple[Self, Dict[Segment, Segment]]]:
        pass

    @abstractmethod
    def support(self, collar: float = 0.) -> Self:
        pass


class ContiguousAnnotationMixin(metaclass=ABCMeta):
    # TODO : figure out if the return mapping still makes sense
    #  (propably not)
    @abstractmethod
    def crop(self,
             support: ContiguousSupport,
             mode: CropMode = 'intersection',
             returns_mapping: bool = False) \
            -> Union[Self, Tuple[Self, Dict[Segment, Segment]]]:
        pass


class SegmentSetMixin(metaclass=ABCMeta):

    @abstractmethod
    def add(self, segment: Segment):
        pass

    @abstractmethod
    def remove(self, segment: Segment):
        pass

    @abstractmethod
    def index(self, segment: Segment) -> int:
        pass

    @abstractmethod
    def overlapping(self, t: float) -> List[Segment]:
        pass
