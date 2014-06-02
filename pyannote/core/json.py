#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import simplejson as json

PYANNOTE_JSON_SEGMENT = 'S'
PYANNOTE_JSON_TIMELINE = 'L'
PYANNOTE_JSON_ANNOTATION = 'A'
PYANNOTE_JSON_TRANSCRIPTION = 'T'

def object_hook(d):
    """
    Usage
    -----
    >>> import simplejson as json
    >>> with open('file.json', 'r') as f:
    ...   json.load(f, object_hook=object_hook)
    """

    from segment import Segment
    from timeline import Timeline
    from annotation import Annotation
    from transcription import Transcription

    if PYANNOTE_JSON_SEGMENT in d:
        return Segment.from_json(d)

    if PYANNOTE_JSON_TIMELINE in d:
        return Timeline.from_json(d)

    if PYANNOTE_JSON_ANNOTATION in d:
        return Annotation.from_json(d)

    if PYANNOTE_JSON_TRANSCRIPTION in d:
        return Transcription.from_json(d)

    return d

def load(path):
    with open(path, 'r') as f:
        data = json.load(f, encoding='utf-8', object_hook=object_hook)
    return data

def dump(data, path):
    # TODO: add pyannote.core version
    with open(path, 'w') as f:
        json.dump(data, f, encoding='utf-8', for_json=True)


