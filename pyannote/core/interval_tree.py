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

from .segment import Segment
import numpy as np


# =====================================================================
# Helper functions
# =====================================================================

def _kth(node, k):
    """Look for kth element of tree rooted at node"""

    if not node:
        raise IndexError("index out of range")

    elif node.left:

        # if there are more than k elements in left child
        # then the kth element should be found in left child
        if k < node.left.metadata.num:
            return _kth(node.left, k)

        # if there are less than k elements in left child
        # then the kth element should be found in right child
        elif k > node.left.metadata.num:
            return _kth(node.right, k - 1 - node.left.metadata.num)

        # if there are exactly k elements in left child
        # then the kth (0-indexed) element is in root node
        else:
            return node.key

    elif k == 0:
        return node.key

    else:
        return _kth(node.right, k - 1)


def _index(node, segment, k):
    """Look for segment in tree rooted at node"""

    if not node:
        raise ValueError("%s is not in timeline" % repr(segment))

    elif segment < node.key:
        return _index(node.left, segment, k)

    elif segment > node.key:
        return _index(node.right,
                      segment,
                      k + 1 + (node.left.metadata.num if node.left else 0))

    else:
        return k + (node.left.metadata.num if node.left else 0)


def _debug(node, depth):
    """Print debug"""

    if node:
        _debug(node.right, depth + 4)
        print(" " * depth + repr(node.key) + " " + repr(node.metadata))
        _debug(node.left, depth + 4)


def _dfs(node):
    """Depth-first search key iterator"""

    if not node:
        return

    for key in _dfs(node.left):
        yield key

    yield node.key

    for key in _dfs(node.right):
        yield key


def _overlapping(node, t):

    segments = []

    if not node:
        return segments

    if node.left:
        if node.left.metadata.extent.overlaps(t):
            segments.extend(_overlapping(node.left, t))

    if node.key.overlaps(t):
        segments.append(node.key)

    if node.right:
        if node.right.metadata.extent.overlaps(t):
            segments.extend(_overlapping(node.right, t))

    return segments


def _intersecting(node, segment):
    """Returns segments intersecting query segment."""

    # empty list of segments
    segments = []

    # search left child if it exists
    if node.left:

        # if left child extent may intersect query `segment`
        # look for intersecting segments in left child
        left_extent = node.left.metadata.extent
        if left_extent.intersects(segment):

            # if left child extent is fully included in query `segment`
            # then every segment intersects query `segment`
            if left_extent in segment:
                segments.extend([key for key in _dfs(node.left)])

            # otherwise, only some of them intersects it
            # and we must look for them
            else:
                segments.extend(_intersecting(node.left, segment))

    # add root segment if it intersects query `segment`
    if node.key.intersects(segment):
        segments.append(node.key)

    # search right child if it exists
    if node.right:

        # if right child extent may intersect query `segment`
        # look for intersecting segments in right child
        right_extent = node.right.metadata.extent
        if right_extent.intersects(segment):

            # if right child extent is fully included in query `segment`
            # then every segment intersects query `segment`
            if right_extent in segment:
                segments.extend([key for key in _dfs(node.right)])

            # otherwise, only some of them intersects it
            # and we must look for them
            else:
                segments.extend(_intersecting(node.right, segment))

    # segments are naturally ordered (depth-first search)
    return segments


def _co_iter(node1, node2):
    """Generator of all pairs of intersecting segments"""

    # stop as soon as one tree is empty
    if not node1 or not node2:
        return

    # stop as as soon as trees do not overlap
    if not node1.metadata.extent.intersects(node2.metadata.extent):
        return

    # ----
    # if left tree #1 is not empty, process it
    # ----
    if node1.left:

        # find overlapping segments in left tree #1 and left tree #2
        for (segment1, segment2) in _co_iter(node1.left, node2.left):
            yield segment1, segment2

        # find segments of left tree #1 overlapping current segment #2
        segment2 = node2.key
        for segment1 in _intersecting(node1.left, segment2):
            yield segment1, segment2

        # find overlapping segments in left tree #1 and right tree #2
        for (segment1, segment2) in _co_iter(node1.left, node2.right):
            yield segment1, segment2

    # ----
    # find segments of tree #2 intersecting current segment #1
    # ----
    segment1 = node1.key

    # find segments of left tree #2 overlapping current segment #1
    if node2.left:
        for segment2 in _intersecting(node2.left, segment1):
            yield segment1, segment2

    # check if current segments #1 and #2 intersects
    segment2 = node2.key
    if segment1.intersects(segment2):
        yield segment1, segment2

    # find segments of right tree #2 overlapping current segment #1
    if node2.right:
        for segment2 in _intersecting(node2.right, segment1):
            yield segment1, segment2

    # ----
    # if right tree #1 is not empty, process it
    # ----
    if node1.right:

        for (segment1, segment2) in _co_iter(node1.right, node2.left):
            yield segment1, segment2

        segment2 = node2.key
        for segment1 in _intersecting(node1.right, segment2):
            yield segment1, segment2

        for (segment1, segment2) in _co_iter(node1.right, node2.right):
            yield segment1, segment2


# =====================================================================
# Banyan SortedSet updator
# =====================================================================

class TimelineUpdator(object):

    class Metadata(object):

        def update(self, key, key_fn, left, right):

            # number of segments in tree
            self.num = 1
            if left:
                self.num += left.num
            if right:
                self.num += right.num

            # extent of tree
            self.extent = key
            if left:
                self.extent = self.extent | left.extent
            if right:
                self.extent = self.extent | right.extent

        def __repr__(self):
            return '{extent: %s, num: %d}' % (repr(self.extent), self.num)

    def length(self):
        if self.root:
            return self.root.metadata.num
        else:
            return 0

    def kth(self, k):
        """Get kth segment"""
        if k < 0:
            return _kth(self.root, self.root.num + k)
        else:
            return _kth(self.root, k)

    def index(self, segment):
        return _index(self.root, segment, 0)

    def extent(self):
        if self.root:
            return self.root.metadata.extent
        else:
            return Segment(start=np.inf, end=-np.inf)

    def debug(self):
        _debug(self.root, 0)

    def overlapping(self, t):
        """Get list of segments overlapping time t"""
        return _overlapping(self.root, t)

    def intersecting(self, segment):
        return _intersecting(self.root, segment)

    def co_iter(self, other):
        for segment, other_segment in _co_iter(self.root, other.root):
            yield segment, other_segment
