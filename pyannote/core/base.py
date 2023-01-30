from abc import ABCMeta, abstractmethod
from typing import Optional, Iterable, Iterator, Tuple, Union, Dict, TYPE_CHECKING, Callable

from typing_extensions import Self

from pyannote.core import Segment
from pyannote.core.utils.types import Support, CropMode

if TYPE_CHECKING:
    from .timeline import Timeline


class BaseSegmentation(metaclass=ABCMeta):
    """Abstract base  class for all segmented annotations"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __nonzero__(self):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __eq__(self, other: Self):
        pass

    @abstractmethod
    def __ne__(self, other: Self):
        pass

    @abstractmethod
    def update(self, other: Self) -> Self:
        pass

    @abstractmethod
    def co_iter(self, other: 'BaseSegmentation') -> Iterator[Tuple[Segment, Segment]]:
        pass

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
    def __contains__(self, included: Union[Segment, 'Timeline']):
        pass

    @abstractmethod
    def empty(self) -> 'Timeline':
        pass

    @abstractmethod
    def copy(self, segment_func: Optional[Callable[[Segment], Segment]] = None) \
            -> Self:
        pass

    @abstractmethod
    def extent(self) -> Segment:
        pass

    @abstractmethod
    def support_iter(self, collar: float = 0.0) -> Iterator[Segment]:
        pass

    @abstractmethod
    def support(self, collar: float = 0.) -> 'Timeline':
        pass

    @abstractmethod
    def duration(self) -> float:
        pass

    @abstractmethod
    def for_json(self):
        pass

    @classmethod
    # TODO
    def from_json(cls, data):
        pass

    @abstractmethod
    def _repr_png_(self):
        pass


class GappedAnnotationMixin(metaclass=ABCMeta):

    @abstractmethod
    def crop_iter(self,
                  support: Support,
                  mode: CropMode = 'intersection',
                  returns_mapping: bool = False) \
            -> Iterator[Union[Tuple[Segment, Segment], Segment]]:
        pass

    @abstractmethod
    def crop(self,
             support: Support,
             mode: CropMode = 'intersection',
             returns_mapping: bool = False) \
            -> Union['Timeline', Tuple['Timeline', Dict[Segment, Segment]]]:
        pass

    @abstractmethod
    def gaps_iter(self, support: Optional[Support] = None) -> Iterator[Segment]:
        pass

    @abstractmethod
    def gaps(self, support: Optional[Support] = None) -> 'Timeline':
        pass

    @abstractmethod
    def extrude(self,
                removed: Support,
                mode: CropMode = 'intersection') -> 'Timeline':
        pass
