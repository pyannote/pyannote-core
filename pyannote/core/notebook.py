#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

from __future__ import unicode_literals

from IPython.core.pylabtools import print_figure
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import itertools
import codecs
import tempfile
import networkx as nx
import numpy as np
import subprocess

from segment import Segment

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Notebook:
    pass

_notebook = Notebook()
_notebook.extent = None
_DEFAULT_NOTEBOOK_WIDTH = 20
_notebook.width = _DEFAULT_NOTEBOOK_WIDTH


def set_notebook_crop(segment=None, margin=0.1):

    if segment is None:
        _notebook.extent = None
    else:
        assert isinstance(segment, Segment)
        assert segment
        duration = segment.duration
        _notebook.extent = Segment(
            segment.start - margin * duration,
            segment.end + margin * duration
        )


def set_notebook_width(inches=None):
    if inches is None:
        _notebook.width = _DEFAULT_NOTEBOOK_WIDTH
    else:
        _notebook.width = inches

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _setup(ylim=None, yaxis=False):
    """Prepare figure"""
    fig, ax = plt.subplots()
    ax.set_xlim(_notebook.extent)
    ax.set_xlabel('Time')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.axes.get_yaxis().set_visible(yaxis)
    return fig, ax


def _render(fig):
    """Render figure as png and return raw image data"""
    # render figure as png
    data = print_figure(fig, 'png')
    # prevent IPython notebook from displaying the figure
    plt.close(fig)
    # return raw image data
    return data


def _draw_segment(ax, segment, y, color, label=None, text=True,
                  boundaries=True):

    # do nothing if segment is empty
    if not segment:
        return

    # draw segment
    ax.hlines(y, segment.start, segment.end, color, lw=1, label=label)
    if boundaries:
        ax.vlines(segment.start, y + 0.05, y - 0.05, color, lw=1)
        ax.vlines(segment.end, y + 0.05, y - 0.05, color, lw=1)

    if label is None:
        return

    # draw label
    if text:
        ax.text(
            segment.middle,
            y + 0.05,
            codecs.encode(unicode(label), 'ascii', 'replace'),
            horizontalalignment='center',
            fontsize=10
        )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def repr_segment(segment):
    """Get `png` data for `segment`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (_notebook.width, 1)

    if not _notebook.extent:
        set_notebook_crop(segment=segment)

    fig, ax = _setup(ylim=(0, 1))

    # segments are vertically centered and blue
    y = 0.5
    color = 'b'
    _draw_segment(ax, segment, y, color)

    data = _render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _y(segments):
    """

    Parameters
    ----------
    segments : iterator
        `Segment` iterator (sorted)

    Returns
    -------
    y : np.array
        y coordinates of each segment

    """

    # up_to stores the largest end time
    # displayed in each line (at the current iteration)
    # (at the beginning, there is only one empty line)
    up_to = [-np.inf]

    # y[k] indicates on which line to display kth segment
    y = []

    for segment in segments:
        # so far, we do not know which line to use
        found = False
        # try each line until we find one that is ok
        for i, u in enumerate(up_to):
            # if segment starts after the previous one
            # on the same line, then we add it to the line
            if segment.start >= u:
                found = True
                y.append(i)
                up_to[i] = segment.end
                break
        # in case we went out of lines, create a new one
        if not found:
            y.append(len(up_to))
            up_to.append(segment.end)

    # from line numbers to actual y coordinates
    y = 1. - 1. / (len(up_to) + 1) * (1 + np.array(y))

    return y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def repr_timeline(timeline):
    """Get `png` data for `timeline`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (_notebook.width, 1)

    if not _notebook.extent and timeline:
        set_notebook_crop(segment=timeline.extent())

    cropped = timeline.crop(_notebook.extent, mode='loose')

    fig, ax = _setup(ylim=(0, 1))

    color = 'b'
    for segment, y in itertools.izip(cropped, _y(cropped)):
        _draw_segment(ax, segment, y, color)

    data = _render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def repr_annotation(annotation):
    """Get `png` data for `annotation`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (_notebook.width, 2)

    if not _notebook.extent:
        set_notebook_crop(segment=annotation.get_timeline().extent())

    cropped = annotation.crop(_notebook.extent, mode='intersection')
    segments = [s for s, _ in cropped.itertracks()]

    # one color per label
    chart = cropped.chart()
    cm = get_cmap('gist_rainbow')
    colors = {
        label: cm(1. * i / len(chart))
        for i, (label, _) in enumerate(chart)
    }

    fig, ax = _setup(ylim=(0, 1))

    for (segment, track, label), y in itertools.izip(
            cropped.itertracks(label=True), _y(segments)):
        color = colors[label]  # color = f(label)
        _draw_segment(ax, segment, y, color, label=label)

    data = _render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def repr_scores(scores):
    """Get `png` data for `scores`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (_notebook.width, 5)

    if not _notebook.extent:
        set_notebook_crop(segment=scores.get_timeline().extent())

    cropped = scores.crop(_notebook.extent, mode='loose')

    cm = get_cmap('gist_rainbow')
    labels = sorted(scores.labels())
    colors = {label: cm(1. * i / len(labels))
              for i, label in enumerate(labels)}

    data = scores.dataframe_.values
    mu = np.nanmean(data)
    sigma = np.nanstd(data)
    ylim = (mu - 3 * sigma, mu + 3 * sigma)

    fig, ax = _setup(yaxis=True, ylim=ylim)

    for segment, track, label, value in cropped.itervalues():
        color = colors[label]
        y = value
        _draw_segment(ax, segment, y, color, label=label, text=False,
                      boundaries=False)

    # for segment, track, label, value in cropped.nbest(1).itervalues():
    #     color = colors[label]
    #     y = value
    #     _draw_segment(ax, segment, y, color, label=label, text=True,
    #                   boundaries=True)

    # get one handle per label and plot the corresponding legend
    H, L = ax.get_legend_handles_labels()
    handles = {}
    for h, l in zip(H, L):
        if l in labels:
            handles[l] = h
    ax.legend([handles[l] for l in sorted(handles)], sorted(handles))

    data = _render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _shorten_long_text(text, max_length=30):
    suffix = "..."
    if len(text) > max_length:
        return text[:max_length - len(suffix)] + suffix
    else:
        return text


def _remove_non_ascii(text):
    # list of characters to remove
    # those characters make resulting SVG invalid
    remove = [u"&", ]
    remove = {ord(r): u"" for r in remove}
    ascii = unicode(codecs.encode(text, 'ascii', 'replace'))
    return ascii.translate(remove)


def _dottable(transcription):
    # create new graph with graphviz/dot layout instructions
    # (e.g. graph orientation, node and edge labels, etc.)

    # create a dumb networkx copy to avoid messing with input transcription
    dottable = transcription.copy()

    # graphviz/dot will display graph from left to right
    dottable.graph['graph'] = {'rankdir': 'LR'}

    # set shape, label and tooltip of each node
    for n in transcription.nodes_iter():

        dottable.node[n] = {
            'label': str(n), 'tooltip': str(n),
            'shape': 'circle' if n.drifting else 'box'
        }
        # 'URL': 'javascript:console.log("{t}")'.format(t=n.T),

    # set edge label

    label_header = (
        "<<table border='0' cellborder='0' cellspacing='0' cellpadding='3'>"
    )
    label_pattern = (
        "<tr>"
        "<td align='left'><b>{name}</b></td>"
        "<td align='left'>{value}</td>"
        "</tr>"
    )
    label_footer = "</table>>"
    tooltip_pattern = "[{name}] {value}"

    for source, target, key, data in transcription.edges_iter(
        keys=True, data=True
    ):
        tooltip = ""
        label = ""
        if data:

            # initialize label table
            label = label_header

            for name, value in data.iteritems():
                # remove non-ascii characters
                name = _remove_non_ascii(name)
                value = _remove_non_ascii(value)
                # shorten long value
                short_value = _shorten_long_text(value)
                # update label and tooltip
                label += label_pattern.format(name=name, value=short_value)
                tooltip += tooltip_pattern.format(name=name, value=value)

            # close label table
            label += label_footer

        dottable[source][target][key] = {
            'label': label,
            'labeltooltip': tooltip,
            'edgetooltip': tooltip,
            'headtooltip': tooltip,
            'tailtooltip': tooltip,
        }

    return dottable


def _write_temporary_dot_file(transcription):
    _, path = tempfile.mkstemp('.dot')
    nx.write_dot(_dottable(transcription), path)
    return path


def repr_transcription(transcription):
    """Get `svg` data for `transcription`"""
    path = _write_temporary_dot_file(transcription)
    data = subprocess.check_output(["dot", "-T", "svg", path])
    return data[data.find("<svg"):]
