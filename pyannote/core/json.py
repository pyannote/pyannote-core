#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2016 CNRS

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

import simplejson as json

PYANNOTE_JSON = 'pyannote'
PYANNOTE_JSON_CONTENT = 'content'


def object_hook(d):
    """
    Usage
    -----
    >>> with open('file.json', 'r') as f:
    ...   json.load(f, object_hook=object_hook)
    """

    if PYANNOTE_JSON in d:
        import pyannote.core
        cls = getattr(pyannote.core, d[PYANNOTE_JSON])
        return cls.from_json(d)

    return d


def load(fp):
    return json.load(fp, encoding='utf-8', object_hook=object_hook)


def loads(s):
    return json.loads(s, encoding='utf-8', object_hook=object_hook)


def load_from(path):
    with open(path, 'r') as fp:
        return load(fp)


def dump(resource, fp):
    json.dump(resource, fp, encoding='utf-8', for_json=True)


def dumps(resource):
    return json.dumps(resource, encoding='utf-8', for_json=True)


def dump_to(resource, path):
    with open(path, 'w') as fp:
        dump(resource, fp)
