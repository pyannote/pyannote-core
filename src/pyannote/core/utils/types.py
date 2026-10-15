from collections.abc import Hashable, Iterator
from typing import Literal, Union

Label = Hashable
Support = Union["Segment", "Timeline"]
LabelGeneratorMode = Literal["int", "string"]
LabelGenerator = Union[LabelGeneratorMode, Iterator[Label]]
TrackName = Union[str, int]
Key = Union["Segment", tuple["Segment", TrackName]]
Resource = Union["Segment", "Timeline", "SlidingWindowFeature", "Annotation"]
CropMode = Literal["intersection", "loose", "strict"]
Alignment = Literal["center", "loose", "strict"]
LabelStyle = tuple[str, int, tuple[float, float, float]]
