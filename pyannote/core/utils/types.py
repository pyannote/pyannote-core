from typing import Hashable, Union, Tuple, Iterator

from typing_extensions import Literal

Label = Hashable
Support = Union['Segment', 'SegmentSet']
ContiguousSupport = Union['Segment', 'ContiguousAnnotationMixin']
LabelGeneratorMode = Literal['int', 'string']
LabelGenerator = Union[LabelGeneratorMode, Iterator[Label]]
TrackName = Union[str, int]
Key = Union['Segment', Tuple['Segment', TrackName]]
TierName = str
TierItemPair = Tuple[TierName, 'Segment']
Resource = Union['Segment', 'Timeline', 'SlidingWindowFeature',
                 'Annotation']
CropMode = Literal['intersection', 'loose', 'strict']
Alignment = Literal['center', 'loose', 'strict']
LabelStyle = Tuple[str, int, Tuple[float, float, float]]
