#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2017 CNRS

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

"""
#############
Visualization
#############
"""

from __future__ import unicode_literals
from __future__ import print_function

try:
    from IPython.core.pylabtools import print_figure
except Exception as e:
    pass
from matplotlib.cm import get_cmap
import six.moves
import tempfile
import networkx as nx
import numpy as np
import subprocess
from itertools import cycle, product, groupby
from .segment import Segment
from .timeline import Timeline
from .annotation import Annotation
from .scores import Scores


class Notebook(object):

    def __init__(self):
        super(Notebook, self).__init__()
        self.reset()

    def reset(self):
        linewidth = [3, 1]
        linestyle = ['solid', 'dashed', 'dotted']

        cm = get_cmap('Set1')
        colors = [cm(1. * i / 8) for i in range(9)]

        self._style_generator = cycle(product(linestyle, linewidth, colors))
        self._style = {None: ('solid', 1, (0.0, 0.0, 0.0))}
        del self.crop
        del self.width

    def crop():
        doc = "The crop property."
        def fget(self):
            return self._crop
        def fset(self, segment):
            self._crop = segment
        def fdel(self):
            self._crop = None
        return locals()
    crop = property(**crop())

    def width():
        doc = "The width property."
        def fget(self):
            return self._width
        def fset(self, value):
            self._width = value
        def fdel(self):
            self._width = 20
        return locals()
    width = property(**width())

    def __getitem__(self, label):
        if label not in self._style:
            self._style[label] = next(self._style_generator)
        return self._style[label]

    def setup(self, ax=None, ylim=(0, 1), yaxis=False, time=True):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.crop)
        if time:
            ax.set_xlabel('Time')
        else:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.axes.get_yaxis().set_visible(yaxis)
        return ax

    def draw_segment(self, ax, segment, y, label=None, boundaries=True):

        # do nothing if segment is empty
        if not segment:
            return

        linestyle, linewidth, color = self[label]

        # draw segment
        ax.hlines(y, segment.start, segment.end, color,
                 linewidth=linewidth, linestyle=linestyle, label=label)
        if boundaries:
            ax.vlines(segment.start, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')
            ax.vlines(segment.end, y + 0.05, y - 0.05,
                      color, linewidth=1, linestyle='solid')

        if label is None:
            return

    def get_y(self, segments):
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


    def __call__(self, resource, time=True, legend=True):

        if isinstance(resource, Segment):
            self.plot_segment(resource, time=time)

        elif isinstance(resource, Timeline):
            self.plot_timeline(resource, time=time)

        elif isinstance(resource, Annotation):
            self.plot_annotation(resource, time=time, legend=legend)

        elif isinstance(resource, Scores):
            self.plot_scores(resource, time=time, legend=legend)


    def plot_segment(self, segment, ax=None, time=True):

        if not self.crop:
            self.crop = segment

        ax = self.setup(ax=ax, time=time)
        self.draw_segment(ax, segment, 0.5)

    def plot_timeline(self, timeline, ax=None, time=True):

        if not self.crop and timeline:
            self.crop = timeline.extent()

        cropped = timeline.crop(self.crop, mode='loose')

        ax = self.setup(ax=ax, time=time)

        for segment, y in six.moves.zip(cropped, self.get_y(cropped)):
            self.draw_segment(ax, segment, y)

        # ax.set_aspect(3. / self.crop.duration)

    def plot_annotation(self, annotation, ax=None, time=True, legend=True):

        if not self.crop:
            self.crop = annotation.get_timeline(copy=False).extent()

        cropped = annotation.crop(self.crop, mode='intersection')
        labels = cropped.labels()
        segments = [s for s, _ in cropped.itertracks()]

        ax = self.setup(ax=ax, time=time)

        for (segment, track, label), y in six.moves.zip(
                cropped.itertracks(label=True), self.get_y(segments)):
            self.draw_segment(ax, segment, y, label=label)

        if legend:
            # this gets exactly one legend handle and one legend label per label
            # (avoids repeated legends for repeated tracks with same label)
            H, L = ax.get_legend_handles_labels()
            HL = groupby(sorted(zip(H, L), key=lambda h_l: h_l[1]),
                         key=lambda h_l: h_l[1])
            H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))
            ax.legend(H, L, bbox_to_anchor=(0, 1), loc=3,
                      ncol=5, borderaxespad=0., frameon=False)

    def plot_scores(self, scores, ax=None, time=True, legend=True):

        if not self.crop:
            self.crop = scores.to_annotation().get_timeline(copy=False).extent()

        cropped = scores.crop(notebook.crop, mode='loose')
        labels = cropped.labels()

        data = scores.dataframe_.values
        m = np.nanmin(data)
        M = np.nanmax(data)
        ylim = (m - 0.1 * (M - m), M + 0.1 * (M - m))

        ax = self.setup(ax=ax, yaxis=True, ylim=ylim, time=time)

        for segment, track, label, value in cropped.itervalues():
            y = value
            self.draw_segment(ax, segment, y, label=label, boundaries=False)

        # ax.set_aspect(6. / ((ylim[1] - ylim[0]) * self.crop.duration))

        if legend:
            # this gets exactly one legend handle and one legend label per label
            # (avoids repeated legends for repeated tracks with same label)
            H, L = ax.get_legend_handles_labels()
            HL = groupby(sorted(zip(H, L), key=lambda h_l: h_l[1]),
                         key=lambda h_l: h_l[1])
            H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))
            ax.legend(H, L, bbox_to_anchor=(0, 1), loc=3,
                      ncol=5, borderaxespad=0., frameon=False)

    def plot_feature(self, feature, ax=None, time=True):

        if not self.crop:
            self.crop = feature.getExtent()

        window = feature.sliding_window
        indices = window.crop(self.crop, mode='loose')
        t = [window[i].middle for i in indices]

        data = np.take(feature.data, indices, axis=0, mode='clip')
        m = np.nanmin(data)
        M = np.nanmax(data)
        ylim = (m - 0.1 * (M - m), M + 0.1 * (M - m))

        ax = self.setup(ax=ax, yaxis=False, ylim=ylim, time=time)
        ax.plot(t, data)

notebook = Notebook()


def repr_segment(segment):
    """Get `png` data for `segment`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 1)
    fig, ax = plt.subplots()
    notebook.plot_segment(segment, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_timeline(timeline):
    """Get `png` data for `timeline`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 1)
    fig, ax = plt.subplots()
    notebook.plot_timeline(timeline, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_annotation(annotation):
    """Get `png` data for `annotation`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_annotation(annotation, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_scores(scores):
    """Get `png` data for `scores`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_scores(scores, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def repr_feature(feature):
    """Get `png` data for `feature`"""
    import matplotlib.pyplot as plt
    figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_feature(feature, ax=ax)
    data = print_figure(fig, 'png')
    plt.close(fig)
    plt.rcParams['figure.figsize'] = figsize
    return data


def _shorten_long_text(text, max_length=30):
    suffix = "..."
    if len(text) > max_length:
        return text[:max_length - len(suffix)] + suffix
    else:
        return text


def _clean_text(text):

    from unidecode import unidecode

    only_ascii = six.u(unidecode(text))

    # remove characters that make resulting SVG invalid
    mapping = {}
    mapping[ord(u'&')] = None
    # mapping[ord(u'"')] = u"'"

    return only_ascii.translate(mapping)


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

            for name, value in six.iteritems(data):
                # remove non-ascii characters
                name = _clean_text(name)
                value = _clean_text(str(value))
                # shorten long value
                short_value = _shorten_long_text(value)
                # update label and tooltip
                label += label_pattern.format(name=name, value=short_value)
                tooltip += tooltip_pattern.format(name=name, value=value)

            # close label table
            label += label_footer

        if not tooltip:
            tooltip = " "

        if not label:
            label = " "

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
    data = subprocess.check_output(["dot", "-T", "svg", path]).decode('ascii')
    return data[data.find("<svg"):]
