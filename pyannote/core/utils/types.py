from typing import Hashable, Union, Tuple, Generator

from typing_extensions import Literal

Label = Hashable
Support = Union['Segment', 'Timeline']
LabelGenerator = Union[str, Generator[Label, None, None]]
TrackName = Union[str, int]
Key = Union['Segment', Tuple['Segment', TrackName]]
Resource = Union['Segment', 'Timeline', 'Score', 'SlidingWindowFeature',
                 'Annotation']
CropMode = Literal['intersection', 'loose', 'strict']
SegmentCropMode = Literal['center', 'loose', 'strict']
LabelStyle = Tuple[str, int, Tuple[float, float, float]]
