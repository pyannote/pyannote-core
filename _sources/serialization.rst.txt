
#############
Serialization
#############

:mod:`pyannote.core.json` provides convenient functions to (de)serialize
`pyannote.core` data structure instances to (from) JSON.

.. ipython::

  In [14]: from pyannote.core import Segment, Timeline, Annotation

  In [13]: import pyannote.core.json

  In [15]: pyannote.core.json.dumps(Segment(0, 1))
  Out[15]: '{"start": 0, "end": 1}'

  In [16]: timeline = Timeline([Segment(0, 1), Segment(1, 3)], uri='my_file')

  In [22]: serialized = pyannote.core.json.dumps(timeline)

  In [23]: deserialized = pyannote.core.json.loads(serialized)

  In [24]: deserialized == timeline
  Out[24]: True

See :mod:`pyannote.core.json` for the complete reference.
