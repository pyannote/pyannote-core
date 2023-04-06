from abc import ABCMeta, abstractmethod
from typing import Optional, Iterator, Tuple, Union, Dict, TYPE_CHECKING, Callable, List, Set, Iterable

from sortedcontainers import SortedList
from typing_extensions import Self

from pyannote.core import Segment
from pyannote.core.utils.types import Support, CropMode, ContiguousSupport, Label

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

    def __nonzero__(self):
        return self.__bool__()

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

    def segments_set(self) -> Set[Segment]:
        # default implementation, may be overriden for better performance
        return set(self.itersegments())

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
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __contains__(self, included: Union[Segment, 'BaseSegmentation']) -> bool:
        # Base implementation, may be overloaded for better performance
        seg_set = self.segments_set()
        if isinstance(included, Segment):
            return included in seg_set
        elif isinstance(included, BaseSegmentation):
            return seg_set.issuperset(included.segments_set())
        else:
            raise ValueError("")

    @abstractmethod
    def empty(self) -> Self:
        pass

    @abstractmethod
    def copy(self) -> Self:
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

# TODO: rename to SegmentSet?
class GappedAnnotationMixin(metaclass=ABCMeta):

    @abstractmethod
    def gaps_iter(self, support: Optional[Support] = None) -> Iterator[Segment]:
        pass

    @abstractmethod
    def gaps(self, support: Optional[Support] = None) -> 'Timeline':
        pass

    def extrude(self,
                removed: Support,
                mode: CropMode = 'intersection') -> Self:
        """Remove segments that overlap `removed` support.

                Parameters
                ----------
                removed : Segment or Timeline
                    If `support` is a `Timeline`, its support is used.
                mode : {'strict', 'loose', 'intersection'}, optional
                    Controls how segments that are not fully included in `removed` are
                    handled. 'strict' mode only removes fully included segments. 'loose'
                    mode removes any intersecting segment. 'intersection' mode removes
                    the overlapping part of any intersecting segment.

                Returns
                -------
                extruded : Timeline
                    Extruded timeline

                Examples
                --------

                >>> timeline = Timeline([Segment(0, 2), Segment(1, 2), Segment(3, 5)])
                >>> timeline.extrude(Segment(1, 2))
                <Timeline(uri=None, segments=[<Segment(0, 1)>, <Segment(3, 5)>])>

                >>> timeline.extrude(Segment(1, 3), mode='loose')
                <Timeline(uri=None, segments=[<Segment(3, 5)>])>

                >>> timeline.extrude(Segment(1, 3), mode='strict')
                <Timeline(uri=None, segments=[<Segment(0, 2)>, <Segment(3, 5)>])>

                """
        if isinstance(removed, Segment):
            removed = Timeline([removed])
        else:
            removed = removed.get_timeline()

        extent_tl = Timeline([self.extent()], uri=self.uri)
        truncating_support = removed.gaps(support=extent_tl)
        # loose for truncate means strict for crop and vice-versa
        if mode == "loose":
            mode = "strict"
        elif mode == "strict":
            mode = "loose"
        return self.crop(truncating_support, mode=mode)

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

    @abstractmethod
    def get_overlap(self) -> 'Timeline':
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
        # TODO: add errors messages explaining why the support isn't of the right type
        pass

    @abstractmethod
    def bisect(self, at: float):
        pass


class PureSegmentationMixin(metaclass=ABCMeta):
    """A segmentation containing _only_ segments"""

    # TODO: add __and__ (defaults to crop intersection, not in place), that only takes objects of Self type?

    # TODO: can actually take any BaseSegmentation for add & remove
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

    @abstractmethod
    def __iter__(self) -> Iterable[Segment]:
        pass


class AnnotatedSegmentationMixin(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self) -> Iterable[Tuple[Segment, Label]]:
        pass