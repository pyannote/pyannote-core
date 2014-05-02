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

from IPython.core.pylabtools import print_figure
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import itertools
import codecs
import numpy as np


from segment import Segment

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Notebook:
	pass

_notebook = Notebook()
_notebook.extent = None
_DEFAULT_NOTEBOOK_WIDTH = 10
_notebook.width = _DEFAULT_NOTEBOOK_WIDTH

def set_notebook_crop(segment=None, margin=0.1):

	if segment is None:
		_notebook.extent = None
	else:
		assert isinstance(segment, Segment)
		assert segment
		duration = segment.duration
		_notebook.extent = Segment(
			segment.start-margin*duration, 
			segment.end+margin*duration
		)

def set_notebook_width(inches=None):
	if inches is None:
		_notebook.width = _DEFAULT_NOTEBOOK_WIDTH
	else:
		_notebook.width = inches

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _setup():
	fig, ax = plt.subplots()
	ax.set_xlim(_notebook.extent)
	ax.set_xlabel('Time')
	ax.set_ylim((0, 1))
	ax.axes.get_yaxis().set_visible(False)
	return fig, ax

def _render(fig):
	data = print_figure(fig, 'png')
	plt.close(fig)
	return data

def _draw_segment(ax, segment, y, color, label=None):
	if segment:
		ax.hlines(y, segment.start, segment.end, color, lw=1)
		ax.vlines(segment.start, y+0.05, y-0.05, color, lw=1)
		ax.vlines(segment.end, y+0.05, y-0.05, color, lw=1)
		if label:
			text = ax.text(segment.middle, y+0.05, codecs.encode(unicode(label), 'ascii', 'replace'),
				horizontalalignment='center', fontsize=10)
			# text.draw()
			# [[x0, y0], [x1, y1]] = text.get_window_extent().get_points()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def repr_segment(segment):
	
	# remember current figure size 
	figsize = plt.rcParams['figure.figsize']
	# and update it for segment display
	plt.rcParams['figure.figsize'] = (_notebook.width, 1)

	if not _notebook.extent:
		set_notebook_crop(segment=segment)

	fig, ax = _setup()
	
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
	y = 1. - 1./(len(up_to)+1) * (1+np.array(y))

	return y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def repr_timeline(timeline):

	# remember current figure size 
	figsize = plt.rcParams['figure.figsize']
	# and update it for segment display
	plt.rcParams['figure.figsize'] = (_notebook.width, 1)

	if not _notebook.extent and timeline:
		set_notebook_crop(segment=timeline.extent())

	cropped = timeline.crop(_notebook.extent, mode='loose')

	fig, ax = _setup()
	
	color = 'b'
	for segment, y in itertools.izip(cropped, _y(cropped)):
		_draw_segment(ax, segment, y, color)

	data = _render(fig)

	# go back to previous figure size
	plt.rcParams['figure.figsize'] = figsize

	return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def repr_annotation(annotation):

	# remember current figure size 
	figsize = plt.rcParams['figure.figsize']
	# and update it for segment display
	plt.rcParams['figure.figsize'] = (_notebook.width, 2)

	if not _notebook.extent:
		set_notebook_crop(segment=annotation.get_timeline().extent())

	cropped = annotation.crop(_notebook.extent, mode='loose')
	segments = [s for s, _ in cropped.itertracks()]

	# one color per label
	chart = cropped.chart()
	cm = get_cmap('gist_rainbow')
	colors = {
		label: cm(1.*i/len(chart))
		for i, (label, _) in enumerate(chart) 
	}

	fig, ax = _setup()

	for (segment, track, label), y in itertools.izip(
		cropped.itertracks(label=True), _y(segments)):
		color = colors[label]  # color = f(label)
		_draw_segment(ax, segment, y, color, label=label)

	data = _render(fig)

	# go back to previous figure size
	plt.rcParams['figure.figsize'] = figsize

	return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

