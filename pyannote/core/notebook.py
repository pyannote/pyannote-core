#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2106 CNRS

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
from __future__ import print_function

from IPython.core.pylabtools import print_figure
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import six.moves
from unidecode import unidecode
import tempfile
import networkx as nx
import numpy as np
import subprocess
from itertools import cycle, product


class Notebook(object):

    def __init__(self):
        super(Notebook, self).__init__()
        self.reset()

    def reset(self):
        linewidth = [2, 4]
        linestyle = ['solid', 'dashed', 'dotted']

        cm = get_cmap('Set1')
        colors = [cm(1. * i / 8) for i in range(9)]

        self._style_generator = cycle(product(linewidth, linestyle, colors))
        self._style = {None: (1, 'solid', (0.0, 0.0, 0.0))}
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

    def setup(self, ylim=None, yaxis=False):
        """Prepare figure"""
        fig, ax = plt.subplots()
        ax.set_xlim(self.crop)
        ax.set_xlabel('Time')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.axes.get_yaxis().set_visible(yaxis)
        return fig, ax

    def render(self, fig):
        """Render figure as png and return raw image data"""
        # render figure as png
        data = print_figure(fig, 'png')
        # prevent IPython notebook from displaying the figure
        plt.close(fig)
        # return raw image data
        return data

    def draw_segment(self, ax, segment, y, label=None, boundaries=True):

        # do nothing if segment is empty
        if not segment:
            return

        linewidth, linestyle, color = self[label]

        # draw segment
        ax.hlines(y, segment.start, segment.end, color,
                 linewidth=linewidth, linestyle=linestyle, label=label)
        if boundaries:
            ax.vlines(segment.start, y + 0.05, y - 0.05,
                      color, linewidth=linewidth, linestyle=linestyle)
            ax.vlines(segment.end, y + 0.05, y - 0.05,
                      color, linewidth=linewidth, linestyle=linestyle)

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

notebook = Notebook()


def repr_segment(segment):
    """Get `png` data for `segment`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (notebook._width, 1)

    if not notebook.crop:
        notebook.crop = segment

    fig, ax = notebook.setup(ylim=(0, 1))

    # segments are vertically centered and blue
    y = 0.5
    notebook.draw_segment(ax, segment, y)

    data = notebook.render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data


def repr_timeline(timeline):
    """Get `png` data for `timeline`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (notebook._width, 1)

    if not notebook.crop and timeline:
        notebook.crop = timeline.extent()

    cropped = timeline.crop(notebook.crop, mode='loose')

    fig, ax = notebook.setup(ylim=(0, 1))

    for segment, y in six.moves.zip(cropped, notebook.get_y(cropped)):
        notebook.draw_segment(ax, segment, y)

    data = notebook.render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def repr_annotation(annotation):
    """Get `png` data for `annotation`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (notebook._width, 2)

    if not notebook.crop:
        notebook.crop = annotation.get_timeline().extent()

    cropped = annotation.crop(notebook.crop, mode='intersection')
    labels = cropped.labels()
    segments = [s for s, _ in cropped.itertracks()]

    fig, ax = notebook.setup(ylim=(0, 1))

    for (segment, track, label), y in six.moves.zip(
            cropped.itertracks(label=True), notebook.get_y(segments)):
        notebook.draw_segment(ax, segment, y, label=label)

    # get one handle per label and plot the corresponding legend
    H, L = ax.get_legend_handles_labels()
    handles = {}
    for h, l in zip(H, L):
        if l in labels:
            handles[l] = h

    ax.legend([handles[l] for l in sorted(handles)], sorted(handles),
              bbox_to_anchor=(0, 1), loc=3,
              ncol=5, borderaxespad=0., frameon=False)

    data = notebook.render(fig)

    # go back to previous figure size
    plt.rcParams['figure.figsize'] = figsize

    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def repr_scores(scores):
    """Get `png` data for `scores`"""

    # remember current figure size
    figsize = plt.rcParams['figure.figsize']
    # and update it for segment display
    plt.rcParams['figure.figsize'] = (notebook._width, 5)

    if not notebook.crop:
        notebook.crop = scores.to_annotation().get_timeline().extent()

    cropped = scores.crop(notebook.crop, mode='loose')
    labels = cropped.labels()

    data = scores.dataframe_.values
    mu = np.nanmean(data)
    sigma = np.nanstd(data)
    ylim = (mu - 3 * sigma, mu + 3 * sigma)

    fig, ax = notebook.setup(yaxis=True, ylim=ylim)

    for segment, track, label, value in cropped.itervalues():
        y = value
        notebook.draw_segment(ax, segment, y, label=label, boundaries=False)

    # get one handle per label and plot the corresponding legend
    H, L = ax.get_legend_handles_labels()
    handles = {}
    for h, l in zip(H, L):
        if l in labels:
            handles[l] = h

    ncol = 5
    ax.legend([handles[l] for l in sorted(handles)], sorted(handles),
              bbox_to_anchor=(0, 1), loc=3,
              ncol=5, borderaxespad=0., frameon=False)

    data = notebook.render(fig)

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


def _clean_text(text):

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
