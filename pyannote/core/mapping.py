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

class NoMatch(object):
    """
    Parameters
    ----------
    format : str, optional
            
    """
    
    nextID = 0
    """
    Keep track of the number of instances since last reset
    """
    
    @classmethod
    def reset(cls):
        """Reset counter"""
        cls.nextID = 0
    
    @classmethod
    def next(cls):
        """Increment & get counter"""
        cls.nextID += 1
        return cls.nextID
    
    def __init__(self, format='NoMatch%03d'):
        super(NoMatch, self).__init__()
        self.ID = NoMatch.next()
        self.format = format
    
    def __str__(self):
        return self.format % self.ID
    
    def __repr__(self):
        return 'Ø'
        
    def __hash__(self):
        return hash(self.ID)
        
    def __eq__(self, other):
        if isinstance(other, NoMatch):
            return self.ID == other.ID
        else:
            return False

class MElement(object):
    
    def __init__(self, modality, element):
        super(MElement, self).__init__()
        self.modality = modality
        self.element = element
    
    def __eq__(self, other):
        return (self.element == other.element) & \
               (self.modality == other.modality)
    
    def __hash__(self):
        return hash(self.element)
    
    def __str__(self):
        return '%s (%s)' % (self.element, self.modality)
    
    def __repr__(self):
        return str(self)

class Mapping(object):
    """Many-to-many label mapping
    
    Parameters
    ----------
    left : str, optional
        Left-hand side modality. Defaults to 'left'.
    right : str
        Right-hand side modality. Defaults to 'right'.
    
    Returns
    -------
    mapping : Mapping
    
    Examples
    --------
    
        >>> mapping = Mapping('speaker', 'face')
        >>> mapping += (('A', 'B', 'C'), ('a', 'b'))
        >>> mapping += (('D', ), ('c', 'd', e))
        >>> print mapping
        (
           A B C --> a b
           D --> c d e
        )
    
    """
    def __init__(self, left="left", right="right"):
        super(Mapping, self).__init__()
        
        # left & right modality
        self.__left = left
        self.__right = right
        
        # left-to-right one-to-many label mapping
        # { left_label --> sorted_right_labels_tuple }
        # 
        self._left_to_right = {}
        
        # right-to-left one-to-many label mapping
        # { right_label --> sorted_left_labels_tuple }
        self._right_to_left = {}
        
        # left-to-right many-to-many label mapping
        # { sorted_left_labels_tuple --> sorted_right_labels_tuple }
        self._many_to_many = {}
    
    def __get_left(self): 
        return self.__left
    left = property(fget=__get_left)
    """Left-hand side modality"""
    
    def __get_right(self): 
        return self.__right
    right = property(fget=__get_right)
    """Right-hand side modality"""
    
    def __get_left_set(self):
        return set(self._left_to_right)
    left_set = property(fget=__get_left_set)
    """Left-hand side set of labels"""
    
    def __get_right_set(self):
        return set(self._right_to_left)
    right_set = property(fget=__get_right_set)
    """Right-hand side set of labels"""
    
    # Normalize left_right argument to 
    # ( tuple(sorted_left_labels), tuple(sorted_right_labels) )
    def _check_and_normalize(self, left_right):
        
        if not isinstance(left_right, (tuple, list)) or len(left_right) != 2:
            raise ValueError('expected 2-tuple of list of labels')
        
        left = left_right[0]
        right = left_right[1]
        
        if left is None:
            left = tuple()
        elif isinstance(left, (list, tuple, set)):
            left = tuple(sorted(left))
        else:
            raise ValueError('expected list, tuple or set for left part.')
        
        if right is None:
            right = tuple()
        elif isinstance(right, (list, tuple, set)):
            right = tuple(sorted(right))
        else:
            raise ValueError('expected list, tuple or set for right part.')
        
        return left, right
    
    def empty(self):
        """Empty copy"""
        M = type(self)(self.left, self.right)
        return M
    
    def copy(self):
        """Duplicate mapping"""
        
        # starts with an empty mapping
        mapping = self.empty()
        
        # add 
        for left, right in self:
            mapping += (left, right)
        return mapping
    
    def __add__(self, other):
        """
        
        Use expression 'mapping + (left_labels, right_labels)'
                    or 'mapping + other_mapping'
        
        Parameters
        ----------
        other : Mapping or couple of labels
        
        Returns
        -------
        mapping : Mapping
        
        Examples
        --------
        
            >>> mapping1 = Mapping()
            >>> mapping1 += (('A', 'B', 'C'), ('a', 'b'))
            >>> mapping2 = Mapping()
            >>> mapping2 += (('D',), ('c', 'd', 'e'))
            >>> mapping = mapping1 + mapping2
            >>> print mapping
            (
               A B C --> a b
               D --> c d e
            )
        
        """
        
        mapping = self.copy()
        mapping += other
        return mapping
    
    def __iadd__(self, other):
        """
        
        Use expression 'mapping += (left_labels, right_labels)'
                    or 'mapping += other_mapping'
                    
        Examples
        --------

            >>> mapping = Mapping()
            >>> mapping += (('A', 'B', 'C'), ('a', 'b'))
            >>> print mapping
            (
               A B C --> a b
            )
            >>> other_mapping = Mapping()
            >>> other_mapping += (('D',), ('c', 'd', 'e'))
            >>> mapping += other_mapping
            >>> print mapping
            (
               A B C --> a b
               D --> c d e
            )
            
        Raises
        ------
        ValueError
        
        """
        
        if isinstance(other, Mapping):
            if other.left != self.left or other.right != self.right:
               raise ValueError('incompatible mapping modalities')
            
            for left, right in other:
                self += (left, right)
            return self
        
        left, right = self._check_and_normalize(other)
        
        already_mapped = set(left) & self.left_set
        if already_mapped:
            already_mapped = already_mapped.pop()
            raise ValueError('%s (%s) is already mapped to %s.' % \
                             (already_mapped, self.left, \
                              self._left_to_right[already_mapped]))
            
        already_mapped = set(right) & self.right_set
        if already_mapped:
            already_mapped = already_mapped.pop()
            raise ValueError('%s (%s) is already mapped to %s.' % \
                             (already_mapped, self.right, \
                              self._right_to_left[already_mapped]))
        
        for label in left:
            self._left_to_right[label] = right
        for label in right:
            self._right_to_left[label] = left
        
        if left == tuple():
            left = NoMatch()
        if right == tuple():
            right = NoMatch()
        self._many_to_many[left] = right
        
        return self
        
    # --- iterator ---
    
    def __iter__(self):
        """Left/right mapping iterator
        
        Examples
        --------
        
            >>> mapping = Mapping()
            >>> mapping += (('A', 'B', 'C'), ('a', 'b'))
            >>> mapping += (('D',), ('c', 'd', 'e'))
            >>> for left, right in mapping:
            ...    print '%s --> %s' % (' '.join(sorted(left)), \
                                        ' '.join(sorted(right)))
            A B C --> a b
            D --> c d e
            
        """
        
        for left, right in self._many_to_many.iteritems():
            
            if isinstance(left, NoMatch):
                left = set([])
            else:
                left = set(left)
            
            if isinstance(right, NoMatch):
                right = set([])
            else:
                right = set(right)
            
            yield left, right
    
    def __contains__(self, label):
        """Use expression 'label in mapping'"""
        return label in self.left_set
    
    def __hash__(self):
        """Use expression hash(mapping)"""
        return hash(tuple(sorted(self.left_set))) + \
               hash(tuple(sorted(self.right_set)))
    
    def __eq__(self, other):
        """Use expression 'mapping == other_mapping'"""
        return self._left_to_right == other._left_to_right and \
               self._right_to_left == other._right_to_left
        
    def to_partition(self):
        
        partition = {}
        C = 0
        for left, right in self:
            partition.update({MElement(self.left, element): C \
                              for element in left})
            partition.update({MElement(self.right, element): C \
                              for element in right})
            C += 1
        return partition
        
    def to_expected_partition(self):
        
        left = self.left_set
        right = self.right_set
        expected = {element:e for e, element in enumerate(left | right)}

        partition = {}
        for element in left:
            partition[MElement(self.left, element)] = expected[element]
        for element in right:
            partition[MElement(self.right, element)] = expected[element]
        
        return partition
        
    def to_dict(self, reverse=False):
        if reverse:
            return {(tuple(left) if left else NoMatch()): \
                    (tuple(right) if right else NoMatch()) \
                    for right, left in self}
        else:
            return {(tuple(left) if left else NoMatch()): \
                    (tuple(right) if right else NoMatch()) \
                    for left, right in self}
    
    def to_expected_dict(self, reverse=False):
        
        left = self.left_set
        right = self.right_set
        both = left & right
        
        expected_dict = {}
        for element in both:
            expected_dict[element] = element
        for element in left-both:
            expected_dict[element] = NoMatch()
        for element in right-both:
            expected_dict[NoMatch()] = element
        
        if reverse:
            return {value:key for key, value in expected_dict.iteritems()}
        else:
            return expected_dict
    
    def __getitem__(self, key):
        return self._left_to_right[key]
    
    def __str__(self):
        """Human-readable representation
        
        Examples
        --------
        
            >>> mapping = Mapping()
            >>> mapping += (('A', 'B', 'C'), ('a', 'b'))
            >>> mapping += (('D',), ('c', 'd', 'e'))
            >>> print mapping
            (
               A B C --> a b
               D --> c d e
            )
        """
        
        string = "(\n"
        for left, right in self:
            left = [str(i) for i in left]
            right = [str(i) for i in right]
            string += '   %s <--> %s\n' % (' '.join(sorted(left)), \
                                           ' '.join(sorted(right)))
        string += ")"
        return string

class ManyToOneMapping(Mapping):
    """Many-to-one mapping"""
    
    def _check_and_normalize(self, mapping):
        """
        Extra verification for many-to-one mapping
        """
        elements1, elements2 = super(ManyToOneMapping, self)\
                               ._check_and_normalize(mapping)
        
        if len(elements2) > 1:
            raise ValueError('Right mapping part (%s) must contain only one element.' % elements2)
        
        return elements1, elements2
    
    @classmethod
    def fromMapping(cls, mapping):
        M = cls(mapping.left, mapping.right)
        for l, r in mapping:
            M += (l, r)
        return M
        
    def __getitem__(self, key):
        right = self._left_to_right[key]
        if right:
            return right[0]
        else:
            return None
            
    def __call__(self, key):
        return self[key]      

class OneToOneMapping(ManyToOneMapping):
    """One-to-one mapping"""
    
    def _check_and_normalize(self, mapping):
        """
        Extra verification for one-to-one mapping
        """
        elements1, elements2 = super(OneToOneMapping, self)._check_and_normalize(mapping)
        
        if len(elements1) > 1:
            raise ValueError('Left mapping part (%s) must contain only one element.' % elements1)
            
        return elements1, elements2
    
    @classmethod
    def fromMapping(cls, mapping):
        M = cls(mapping.left, mapping.right)
        for l, r in mapping:
            M += (l, r)
        return M
    
    def to_dict(self, reverse=False, single=False):
        D = super(OneToOneMapping, self).to_dict(reverse=reverse)
        if single:
            d = {left if isinstance(left, NoMatch) else left[0]: \
                 right if isinstance(right, NoMatch) else right[0] \
                 for left, right in D.iteritems()}
            return d
        else:
            return D


if __name__ == "__main__":
    import doctest
    doctest.testmod()

