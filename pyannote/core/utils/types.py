from typing import Hashable, Union, Tuple, Iterable, Generator

Label = Hashable
Support = Union['Segment', 'Timeline']
Key = Union['Segment', Tuple['Segment', str]]
LabelGenerator = Union[str, Generator[Label, None, None]]
