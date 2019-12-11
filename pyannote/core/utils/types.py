from typing import Hashable, Union, Tuple, Generator

Label = Hashable
Support = Union['Segment', 'Timeline']
LabelGenerator = Union[str, Generator[Label, None, None]]
TrackName = Union[str, int]
Key = Union['Segment', Tuple['Segment', TrackName]]
Resource = Union['Segment', 'Timeline', 'Score', 'SlidingWindowFeature']

