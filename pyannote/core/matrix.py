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

import numpy as np
import pandas
from segment import SEGMENT_PRECISION

class LabelMatrix(object):

    def __init__(self, data=None, dtype=None, rows=None, columns=None):
        super(LabelMatrix, self).__init__()
        if data is None and dtype is None:
            dtype = np.float
        self.df = pandas.DataFrame(
            data=data, dtype=dtype, index=rows, columns=columns)

    def __setitem__(self, (row, col), value):
        self.df = self.df.set_value(row, col, value)
        return self

    def __getitem__(self, (row, col)):
        return self.df.at[row, col]

    def get_rows(self):
        return list(self.df.index)

    def get_columns(self):
        return list(self.df.columns)

    def __get_shape(self):
        return self.df.shape
    shape = property(fget=__get_shape)

    def __nonzero__(self):
        N, M = self.df.shape
        return N*M != 0

    def itervalues(self):
        for row in self.get_rows():
            for col in self.get_columns():
                val = self.df.at[row, col]
                if not np.isnan(val):
                    yield row, col, val

    def iter_values(self):
        for row in self.get_rows():
            for col in self.get_columns():
                val = self.df.at[row, col]
                if not np.isnan(val):
                    yield row, col, val

    def argmax(self, axis=None):
        """
        Labels of the maximum values along an axis.

        Parameters
        ----------
        axis : int, optional
            By default, labels are into the whole matrix, otherwise
            along the specified axis (rows or columns)

        Returns
        -------
        label_dict : dictionary of labels
            Dictionary of labels into the matrix.
            {col_label : max_row_label} if axis == 0
            {row_label : max_col_label} if axis == 1
            {max_row_label : max_col_label} if axis == None
        """

        if axis == 0:
            return {c: r
                    for (c, r) in self.df.idxmax(axis=axis).iteritems()}

        elif axis == 1:
            return {r: c
                    for (r, c) in self.df.idxmax(axis=axis).iteritems()}

        else:
            values = [
                (_r, _c, self.df.loc[_r, _c])
                for (_c, _r) in self.df.idxmax(axis=0).iteritems()
            ]
            r, c, _ = sorted(values, key=lambda v: v[2])[-1]
            return {r: c}

    def __neg__(self):
        negated = LabelMatrix()
        negated.df = -self.df
        return negated

    def argmin(self, axis=None):
        """
        Labels of the minimum values along an axis.

        Parameters
        ----------
        axis : int, optional
            By default, labels are into the whole matrix, otherwise
            along the specified axis (rows or columns)

        Returns
        -------
        label_dict : dictionary of labels
            Dictionary of labels into the matrix.
            {col_label : max_row_label} if axis == 0
            {row_label : max_col_label} if axis == 1
            {max_row_label : max_col_label} if axis == None
        """

        return (-self).argmax(axis=axis)

    def __get_T(self):
        transposed = LabelMatrix()
        transposed.df = self.df.T
        return transposed
    T = property(fget=__get_T)

    def remove_column(self, col):
        del self.df[col]
        return self

    def remove_row(self, row):
        df = self.df.T
        del df[row]
        self.df = df.T
        return self

    def copy(self):
        copied = LabelMatrix()
        copied.df = self.df.copy()
        return copied

    def subset(self, rows=None, columns=None):

        if rows is None:
            rows = set(self.get_rows())

        if columns is None:
            columns = set(self.get_columns())

        remove_rows = set(self.get_rows()) - rows
        remove_columns = set(self.get_columns()) - columns

        copied = self.copy()
        for row in remove_rows:
            copied = copied.remove_row(row)
        for col in remove_columns:
            copied = copied.remove_column(col)

        return copied

    def __gt__(self, value):
        compared = LabelMatrix()
        compared.df = self.df > value
        return compared

    def __str__(self):
        return str(self.df)

    def _repr_html_(self):
        return self.df._repr_html_()


def get_cooccurrence_matrix(R, C):

    # initialize label matrix with zeros
    rows = R.labels()
    cols = C.labels()
    K = np.zeros((len(rows), len(cols)), dtype=np.float)
    M = LabelMatrix(data=K, rows=rows, columns=cols)
    
    # loop on intersecting tracks
    for (r_segment, r_track), (c_segment, c_track) in R.co_iter(C):
        # increment 
        r_label = R[r_segment, r_track]
        c_label = C[c_segment, c_track]
        duration = (r_segment & c_segment).duration
        M[r_label, c_label] += duration

    return M



def get_tfidf_matrix(words, documents, idf=True, log=False):
    """Term Frequency Inverse Document Frequency (TF-IDF) confusion matrix

    C[i, j] = TF(i, j) x IDF(i) where
        - documents are J labels
        - words are co-occurring I labels

                  duration of word i in document j         confusion[i, j]
    TF(i, j) = --------------------------------------- = -------------------
               total duration of I words in document j   sum confusion[:, j]

                        number of J documents
    IDF(i) = ----------------------------------------------
             number of J documents co-occurring with word i

                      Nj
           = -----------------------
             sum confusion[i, :] > 0

    Parameters
    ---------
    words : :class:`pyannote.base.annotation.Annotation`
        Every label occurrence is considered a word
        (weighted by the duration of the segment)
    documents : :class:`pyannote.base.annotation.Annotation`
        Every label is considered a document.
    idf : bool, optional
        If `idf` is set to True, returns TF x IDF.
        Otherwise, returns TF. Default is True
    log : bool, optional
        If `log` is True, returns TF x log IDF
    """
    M = get_cooccurrence_matrix(words, documents)
    Nw, Nd = M.shape

    if Nd == 0:
        return M

    K = M.df.values
    rows = M.get_rows()
    cols = M.get_columns()

    # total duration of all words cooccurring with each document
    # np.sum(self.M, axis=0)[j] = 0 ==> self.M[i, j] = 0 for all i
    # so we can safely use np.maximum(1e-3, ...) to avoid DivideByZero
    tf = K / np.tile(
        np.maximum(SEGMENT_PRECISION, np.sum(K, axis=0)),
        (Nw, 1)
    )

    # use IDF only if requested (default is True ==> use IDF)
    if idf:

        # number of documents cooccurring with each word
        # np.sum(self.M > 0, axis=1)[i] = 0 ==> tf[i, j] = 0 for all i
        # and therefore tf.idf [i, j ] = 0
        # so we can safely use np.maximum(1, ...) to avoid DivideByZero
        idf = np.tile(
            float(Nd) / np.maximum(1, np.sum(K > 0, axis=1)),
            (Nd, 1)).T

        # use log only if requested (defaults is False ==> do not use log)
        if log:
            idf = np.log(idf)

    else:
        idf = 1.

    return LabelMatrix(data=tf * idf, rows=rows, columns=cols)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
