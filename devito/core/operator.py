from __future__ import absolute_import

from devito.core.autotuning import autotune
from devito.cgen_utils import printmark
from devito.exceptions import InvalidOperator
from devito.ir.equations import Eq
from devito.ir.iet import List, Transformer, filter_iterations, retrieve_iteration_tree
from devito.operator import OperatorRunnable
from devito.symbolics import indexify, retrieve_indexed, q_affine
from devito.tools import flatten

__all__ = ['Operator']


class OperatorCore(OperatorRunnable):

    def _specialize_exprs(self, expressions, subs):
        """
        Lower ``expressions`` by: ::

            * Performing indexification (:class:`Function` --> :class:`Indexed`).
            * Applying any user-provided substitution rules.
            * Translating all array accesses by a certain quantity so that they
              become relative to the domain region.

        The latter task is necessary to index into the right memory locations and
        requires some thought.
        Ideally, adding the extent of the left halo+padding region to each array
        access should suffice. There is, however, a complication. Along some
        dimensions, the array accesses may be shifted w.r.t. to the halo region.
        Consider for example the function `u(t, x, y)`, with halo+padding given
        by ((2, 0), (3, 3), (3, 3)) -- see also :meth:`Function._offset_domain`
        for more info about the semantics of the halo and padding tuples. In an
        equation, the array accesses `u[t-1, x, y]`, `u[t, x, y]`, and
        `u[t+1, x, y]` may be used. The presence of `t-1` and `t+1` suggests that
        the three array accesses along `t` are shifted by 1 from the bottom of
        the allocated region, since the smallest index access (-1, from t-1 with
        t=0) would not touch the very first memory location along `t` in an
        iteration space starting from 0. Analogously, the largest index access
        (t_ext+1, from t+1 with t=t_ext) would end up performing an out-of-bounds
        access, as the right halo is 0 along `t`. Hence, instead of adding 2
        to each index function along `t` (the extent of the left halo+padding
        region), we should add 2-1; that is, we should subtract the shifting
        inferred from the encountered array accessed.

        :raises InvalidOperator: if incompatible shiftings along a certain
                                 dimension, perhaps induced by different
                                 :class:`Function`s, are found. This suggests
                                 an issue in the expression specification provided
                                 by the user.
        """
        # Indexification
        expressions = [indexify(i) for i in expressions]

        # Apply user-provided substitution rules
        if subs is not None:
            expressions = [i.xreplace(subs) for i in expressions]

        # Calculate shifting along each dimension
        constraints = {}
        for e in expressions:
            indexed = e.lhs
            f = indexed.base.function
            if not f.is_SymbolicFunction:
                # Not user-provided tensor data, nothing to do
                continue
            for i, d, gap in zip(indexed.indices, f.dimensions, f._offset_domain):
                if not q_affine(i, d):
                    # Sparse iteration, no check possible
                    continue
                shift = i - d
                if not shift.is_Number:
                    raise InvalidOperator("Array access `%s` in %s is not a "
                                          "translated identity function" % (i, indexed))
                if shift != 0 and shift != constraints.setdefault(d, shift):
                    raise InvalidOperator("Array access `%s` in %s with halo %s "
                                          "has incompatible shift %d (expected %d)"
                                          % (i, indexed, gap, shift, constraints[d]))

        # Translate array accesses (halo + shift)
        mapper = {}
        for e in expressions:
            for indexed in retrieve_indexed(e):
                f = indexed.base.function
                if not f.is_SymbolicFunction:
                    continue
                subs = {i: i + gap.left - constraints.get(d, 0) for i, d, gap in
                        zip(indexed.indices, f.dimensions, f._offset_domain)}
                mapper[indexed] = indexed.xreplace(subs)

        # Finally translate the expressions
        expressions = [e.xreplace(mapper) for e in expressions]

        # Lower to /ir.Eq/, thus associating data and iteration space
        expressions = [Eq(i) for i in expressions]

        return expressions

    def _autotune(self, arguments):
        """
        Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use.
        """
        if self.dle_flags.get('blocking', False):
            return autotune(self, arguments, self.dle_arguments)
        else:
            return arguments


class OperatorDebug(OperatorCore):
    """
    Decorate the generated code with useful print statements.
    """

    def __init__(self, expressions, **kwargs):
        super(OperatorDebug, self).__init__(expressions, **kwargs)
        self._includes.append('stdio.h')

        # Minimize the trip count of the sequential loops
        iterations = set(flatten(retrieve_iteration_tree(self.body)))
        mapper = {i: i._rebuild(limits=(max(i.offsets) + 2))
                  for i in iterations if i.is_Sequential}
        self.body = Transformer(mapper).visit(self.body)

        # Mark entry/exit points of each non-sequential Iteration tree in the body
        iterations = [filter_iterations(i, lambda i: not i.is_Sequential, 'any')
                      for i in retrieve_iteration_tree(self.body)]
        iterations = [i[0] for i in iterations if i]
        mapper = {t: List(header=printmark('In nest %d' % i), body=t)
                  for i, t in enumerate(iterations)}
        self.body = Transformer(mapper).visit(self.body)


class Operator(object):

    def __new__(cls, *args, **kwargs):
        cls = OperatorDebug if kwargs.pop('debug', False) else OperatorCore
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj
