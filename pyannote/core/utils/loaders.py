#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from typing import Text

import pandas as pd

from pyannote.core import Segment, Timeline, Annotation

DatabaseName = Text
PathTemplate = Text


def load_rttm(file_rttm):
    """Load RTTM file

    Parameter
    ---------
    file_rttm : `str`
        Path to RTTM file.

    Returns
    -------
    annotations : `dict`
        Speaker diarization as a {uri: pyannote.core.Annotation} dictionary.
    """

    names = [
        "NA1",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_rttm,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )

    annotations = dict()
    for uri, turns in data.groupby("uri"):
        annotation = Annotation(uri=uri)
        for i, turn in turns.iterrows():
            segment = Segment(turn.start, turn.start + turn.duration)
            annotation[segment, i] = turn.speaker
        annotations[uri] = annotation

    return annotations


def load_mdtm(file_mdtm):
    """Load MDTM file

    Parameter
    ---------
    file_mdtm : `str`
        Path to MDTM file.

    Returns
    -------
    annotations : `dict`
        Speaker diarization as a {uri: pyannote.core.Annotation} dictionary.
    """

    names = ["uri", "NA1", "start", "duration", "NA2", "NA3", "NA4", "speaker"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_mdtm,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )

    annotations = dict()
    for uri, turns in data.groupby("uri"):
        annotation = Annotation(uri=uri)
        for i, turn in turns.iterrows():
            segment = Segment(turn.start, turn.start + turn.duration)
            annotation[segment, i] = turn.speaker
        annotations[uri] = annotation

    return annotations


def load_uem(file_uem):
    """Load UEM file

    Parameter
    ---------
    file_uem : `str`
        Path to UEM file.

    Returns
    -------
    timelines : `dict`
        Evaluation map as a {uri: pyannote.core.Timeline} dictionary.
    """

    names = ["uri", "NA1", "start", "end"]
    dtype = {"uri": str, "start": float, "end": float}
    data = pd.read_csv(file_uem, names=names, dtype=dtype, delim_whitespace=True)

    timelines = dict()
    for uri, parts in data.groupby("uri"):
        segments = [Segment(part.start, part.end) for i, part in parts.iterrows()]
        timelines[uri] = Timeline(segments=segments, uri=uri)

    return timelines


def load_lab(path, uri: str = None) -> Annotation:
    """Load LAB file

    Parameter
    ---------
    file_lab : `str`
        Path to LAB file

    Returns
    -------
    data : `pyannote.core.Annotation`
    """

    names = ["start", "end", "label"]
    dtype = {"start": float, "end": float, "label": str}
    data = pd.read_csv(path, names=names, dtype=dtype, delim_whitespace=True)

    annotation = Annotation(uri=uri)
    for i, turn in data.iterrows():
        segment = Segment(turn.start, turn.end)
        annotation[segment, i] = turn.label

    return annotation


def load_lst(file_lst):
    """Load LST file

    LST files provide a list of URIs (one line per URI)

    Parameter
    ---------
    file_lst : `str`
        Path to LST file.

    Returns
    -------
    uris : `list`
        List or uris
    """

    with open(file_lst, mode="r") as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]


def load_mapping(mapping_txt):
    """Load mapping file

    Parameter
    ---------
    mapping_txt : `str`
        Path to mapping file

    Returns
    -------
    mapping : `dict`
        {1st field: 2nd field} dictionary
    """

    with open(mapping_txt, mode="r") as fp:
        lines = fp.readlines()

    mapping = dict()
    for line in lines:
        key, value, *left = line.strip().split()
        mapping[key] = value

    return mapping
