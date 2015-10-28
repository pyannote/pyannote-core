#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2015 CNRS

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

import six
import networkx as nx
import itertools
from networkx.readwrite.json_graph import node_link_data, node_link_graph

from .time import T, TStart, TEnd
from .segment import Segment
from .json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT
from .util import pairwise


class Transcription(nx.MultiDiGraph):
    """Transcription stored as annotation graph"""

    def __init__(self, graph=None, **attrs):
        super(Transcription, self).__init__(data=graph)
        self.graph.update(attrs)

    def drifting(self):
        """Get list of drifting times"""
        return [n for n in self if n.drifting]

    def anchored(self):
        """Get list of anchored times"""
        return [n for n in self if n.anchored]

    def add_edge(self, t1, t2, key=None, attr_dict=None, **attrs):
        """Add annotation to the graph between times t1 and t2

        Parameters
        ----------
        t1, t2: float, str or None
        data : dict, optional
            {annotation_type: annotation_value} dictionary

        Example
        -------
        >>> G = Transcription()
        >>> G.add_edge(T(1.), T(), speaker='John', 'speech'='Hello world!')
        """
        t1 = T(t1)
        t2 = T(t2)

        # make sure Ts are connected in correct chronological order
        if t1.anchored and t2.anchored:
            assert t1 <= t2

        super(Transcription, self).add_edge(
            t1, t2, key=key, attr_dict=attr_dict, **attrs)

    def relabel_drifting_nodes(self, mapping=None):
        """Relabel drifting nodes

        Parameters
        ----------
        mapping : dict, optional
            A dictionary with the old labels as keys and new labels as values.

        Returns
        -------
        g : Transcription
            New annotation graph
        mapping : dict
            A dictionary with the new labels as keys and old labels as values.
            Can be used to get back to the version before relabelling.
        """

        if mapping is None:
            old2new = {n: T() for n in self.drifting()}
        else:
            old2new = dict(mapping)

        new2old = {new: old for old, new in six.iteritems(old2new)}
        return nx.relabel_nodes(self, old2new, copy=True), new2old

    def crop(self, source, target=None):
        """Get minimum subgraph between source time and target time

        Parameters
        ----------
        source : Segment
        target : float or str, optional

        Returns
        -------
        g : Transcription
            Sub-graph between source and target
        """

        if isinstance(source, Segment):
            source, target = source.start, source.end

        source = T(source)
        target = T(target)

        # sorted list of anchored times will be needed later
        # make sure it is computed only once
        if source.anchored or target.anchored:
            anchored = sorted(self.anchored())

        # ~~~ from_source = set of nodes reachable from source ~~~~~~~~~~~~~~~~

        # source is drifting
        if source.drifting:

            if source not in self:
                raise ValueError(
                    'Drifting time %s is not in the transcription.' % source)
            else:
                from_source = {source} | nx.algorithms.descendants(self, source)

        # source is anchored
        else:

            # if source is in graph, then it is easy
            if source in self:
                from_source = {source} | nx.algorithms.descendants(self, source)

            # if source is not in graph,
            # find anchored time just before source
            else:
                if source < anchored[0]:
                    from_source = set(self)  # take no risk!
                else:
                    before = [n for n in anchored if n <= source][-1]
                    from_source = {before} | nx.algorithms.descendants(self, before)

        # ~~~ to_target = set of nodes from which target is reachable ~~~~~~~~~

        # target is drifting
        if target.drifting:

            if target not in self:
                raise ValueError(
                    'Drifting time %s is not in the transcription.' % target)
            else:
                to_target = {target} | nx.algorithms.ancestors(self, target)

        else:

            # if target is in graph, then it is easy
            if target in self:
                to_target = {target} | nx.algorithms.ancestors(self, target)

            # if target is not in graph,
            # find anchored time just after target
            else:
                if target > anchored[-1]:
                    to_target = set(self)  # take no risk!
                else:
                    after = [n for n in anchored if n >= target][0]
                    to_target = {after} | nx.algorithms.ancestors(self, after)

        # union of source, target and source-to-target paths
        nbunch = from_source & to_target

        return self.subgraph(nbunch)

    # =========================================================================

    def _merge(self, drifting_t, another_t):
        """Helper function to merge `drifting_t` with `another_t`

        Assumes that both `drifting_t` and `another_t` exists.
        Also assumes that `drifting_t` is an instance of `TFloating`
        (otherwise, this might lead to weird graph configuration)

        Parameters
        ----------
        drifting_t :
            Existing drifting time in graph
        another_t :
            Existing time in graph
        """
        # drifting_t and another_t must exist in graph

        # add a (t --> another_t) edge for each (t --> drifting_t) edge
        for t, _, key, data in self.in_edges_iter(
            nbunch=[drifting_t], data=True, keys=True
        ):
            # use lowest unused integer in case this key already exists
            if self.has_edge(t, another_t, key=key):
                key = None
            self.add_edge(t, another_t, key=key, attr_dict=data)

        # add a (another_t --> t) edge for each (drifting_t --> t) edge
        for _, t, key, data in self.edges_iter(
            nbunch=[drifting_t], data=True, keys=True
        ):
            # use lowest unused integer in case this key already exists
            if self.has_edge(another_t, t, key=key):
                key = None
            self.add_edge(another_t, t, key=key, attr_dict=data)

        # remove drifting_t node (as it was replaced by another_t)
        self.remove_node(drifting_t)

    def anchor(self, drifting_t, anchored_t):
        """
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        o -- [ D ] -- o  ==>  o -- [ A ] -- o

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Anchor `drifting_t` at `anchored_t`

        Parameters
        ----------
        drifting_t :
            Drifting time to anchor
        anchored_t :
            When to anchor `drifting_t`

        """

        drifting_t = T(drifting_t)
        anchored_t = T(anchored_t)

        assert (drifting_t in self) and (drifting_t.drifting)
        assert anchored_t.anchored

        if anchored_t not in self:
            self.add_node(anchored_t)

        self._merge(drifting_t, anchored_t)

    def align(self, one_t, another_t):
        """
        Align two (potentially drifting) times
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        o -- [ F ] -- o      o          o
                               ⟍     ⟋
                        ==>     [ F ]
                               ⟋     ⟍
        o -- [ f ] -- o      o          o

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Parameters
        ----------
        one_t, another_t
            Two times to be aligned.

        Notes
        -----
        * If both `one_t` and  `another_t` are drifting, the resulting graph
        will no longer contain `one_t`.
        * In case `another_t` is anchored, `align` is equivalent to `anchor`.
        * `one_t` and `another_t` cannot be both anchored.

        """

        one_t = T(one_t)
        another_t = T(another_t)

        assert one_t in self
        assert another_t in self

        # first time is drifting
        if one_t.drifting:
            self._merge(one_t, another_t)

        # second time is drifting
        elif another_t.drifting:
            self._merge(another_t, one_t)

        # both times are anchored --> FAIL
        else:
            raise ValueError(
                'Cannot align two anchored times')

    # =========================================================================

    def pre_align(self, t1, t2, t):
        """
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        p -- [ t1 ]       p         [ t1 ]
                            ⟍     ⟋
                     ==>     [ t ]
                            ⟋     ⟍
        p' -- [ t2 ]      p'        [ t2 ]

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """

        t1 = T(t1)
        t2 = T(t2)
        t = T(t)

        # make sure --[t1] incoming edges are empty
        # because they're going to be removed afterwards,
        # and we don't want to loose data
        pred1 = self.predecessors(t1)
        for p in pred1:
            for key, data in six.iteritems(self[p][t1]):
                assert not data

        # make sure --[t2] incoming edges are empty
        # (for the same reason...)
        pred2 = self.predecessors(t2)
        for p in pred2:
            for key, data in six.iteritems(self[p][t2]):
                assert not data

        # let's get started (remove all incoming edges)
        for p in pred1:
            for key in list(self[p][t1]):
                self.remove_edge(p, t1, key=key)
        for p in pred2:
            for key in list(self[p][t2]):
                self.remove_edge(p, t2, key=key)

        for p in set(pred1) | set(pred2):
            self.add_edge(p, t)

        self.add_edge(t, t1)
        self.add_edge(t, t2)

    def post_align(self, t1, t2, t):
        """
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        [ t1 ] -- s       [ t1 ]         s
                                ⟍     ⟋
                     ==>         [ t ]
                                ⟋     ⟍
        [ t2 ] -- s'      [ t2 ]        s'

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """

        t1 = T(t1)
        t2 = T(t2)
        t = T(t)

        # make sure [t1]-- outgoing edges are empty
        # because they're going to be removed afterwards,
        # and we don't want to loose data
        succ1 = self.successors(t1)
        for s in succ1:
            for key, data in six.iteritems(self[t1][s]):
                assert not data

        # make sure --[t2] outgoing edges are empty
        # (for the same reason...)
        succ2 = self.successors(t2)
        for s in succ2:
            for key, data in six.iteritems(self[t2][s]):
                assert not data

        # let's get started (remove all outgoing edges)
        for s in succ1:
            for key in list(self[t1][s]):
                self.remove_edge(t1, s, key=key)
        for s in succ2:
            for key in list(self[t2][s]):
                self.remove_edge(t2, s, key=key)

        for s in set(succ1) | set(succ2):
            self.add_edge(t, s)

        self.add_edge(t1, t)
        self.add_edge(t2, t)

    # =========================================================================

    def ordering_graph(self):
        """Ordering graph

        t1 --> t2 in the ordering graph indicates that t1 happens before t2.
        A missing edge simply means that it is not clear yet.

        """

        g = nx.DiGraph()

        # add times
        for t in self.nodes_iter():
            g.add_node(t)

        # add existing edges
        for t1, t2 in self.edges_iter():
            g.add_edge(t1, t2)

        # connect every pair of anchored times
        anchored = sorted(self.anchored())
        for t1, t2 in itertools.combinations(anchored, 2):
            g.add_edge(t1, t2)

        # connect every time with its sucessors
        _g = g.copy()
        for t1 in _g:
            for t2 in set([target for (_, target) in nx.bfs_edges(_g, t1)]):
                g.add_edge(t1, t2)

        return g

    def temporal_sort(self):
        """Get nodes sorted in temporal order

        Remark
        ------
        This relies on a combination of temporal ordering of anchored times
        and topological ordering for drifting times.
        To be 100% sure that one drifting time happens before another time,
        check the ordering graph (method .ordering_graph()).
        """

        g = nx.DiGraph()

        # add times
        for t in self.nodes_iter():
            g.add_node(t)

        # add existing edges
        for t1, t2 in self.edges_iter():
            g.add_edge(t1, t2)

        # connect pairs of consecutive anchored times
        anchored = sorted(self.anchored())
        for t1, t2 in pairwise(anchored):
            g.add_edge(t1, t2)

        return nx.topological_sort(g)

    # =========================================================================

    def ordered_edges_iter(self, nbunch=None, data=False, keys=False):
        """Return an iterator over the edges in temporal order.

        Ordered edges are returned as tuples with optional data and keys
        in the order (t1, t2, key, data).

        Parameters
        ----------
        nbunch : iterable container, optional (default= all nodes)
            A container of nodes. The container will be iterated
            through once.
        data : bool, optional (default=False)
            If True, return edge attribute dict with each edge.
        keys : bool, optional (default=False)
            If True, return edge keys with each edge.

        Returns
        -------
        edge_iter : iterator
            An iterator of (u,v), (u,v,d) or (u,v,key,d) tuples of edges.

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For the same reason you should not completely trust temporal_sort,
        use ordered_edges_iter with care.
        """

        # start by sorting nodes in temporal order
        nodes = self.temporal_sort()

        # only keep nbunch subset (while preserving the order)
        if nbunch:
            nbunch = list(nbunch)
            nodes = [n for n in nodes if n in nbunch]

        # iterate over edges using temporal order
        return self.edges_iter(nbunch=nodes, data=data, keys=keys)

    # =========================================================================

    def timerange(self, t1, t2, inside=True, sort=None):
        """Infer edge timerange from graph structure

        a -- ... -- [ t1 ] -- A -- ... -- B -- [ t2 ] -- ... -- b

        ==> [a, b] (inside=False) or [A, B] (inside=True)

        Parameters
        ----------
        t1, t2 : anchored or drifting times
        inside : boolean, optional

        Returns
        -------
        segment : Segment
        """

        t1 = T(t1)
        t2 = T(t2)

        # in case it is not provided, compute temporal sort
        if sort is None:
            sort = self.temporal_sort()

        # if edge start is anchored, use it as start time
        if t1.anchored:
            start = t1

        # otherwise, look for the closest anchored time in temporal order:
        # - just after if inside is True
        # - just before otherwise
        else:
            start = None
            # find time index in temporal sort
            istart = sort.index(t1)
            # search just before or just after depending on 'inside' value
            search = sort[istart+1:] if inside else sort[istart-1::-1]
            for t in search:
                if t.anchored:
                    start = t
                    break
            # if we could not find any anchored time
            # use document end of start depending on 'inside' value
            if start is None:
                start = TEnd if inside else TStart

        # same treatment for the other end of edge
        if t2.anchored:
            end = t2
        else:
            end = None
            iend = sort.index(t2)
            search = sort[iend-1::-1] if inside else sort[iend+1:]
            for t in search:
                if t.anchored:
                    end = t
                    break
            if end is None:
                end = TStart if inside else TEnd

        # return a 'Segment'
        return Segment(start=start, end=end)

    # =========================================================================

    def for_json(self):

        data = {PYANNOTE_JSON: self.__class__.__name__}
        data[PYANNOTE_JSON_CONTENT] = node_link_data(self)
        return data

    @classmethod
    def from_json(cls, data):
        graph = node_link_graph(data[PYANNOTE_JSON_CONTENT])
        mapping = {node: T(node) for node in graph}
        graph = nx.relabel_nodes(graph, mapping)
        return cls(graph=graph, **graph.graph)

    # === IPython Notebook displays ===========================================

    def _repr_svg_(self):
        from .notebook import repr_transcription
        return repr_transcription(self)
