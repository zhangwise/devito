from __future__ import absolute_import

from collections import OrderedDict
from operator import attrgetter

import cgen as c
import numpy as np
from sympy import Symbol

from devito.cgen_utils import ccode
from devito.dse import as_symbol
from devito.dle import retrieve_iteration_tree, filter_iterations
from devito.dle.backends import AbstractRewriter, dle_pass, complang_ALL
from devito.interfaces import ScalarFunction
from devito.nodes import Denormals, Expression, FunCall, Function, List
from devito.tools import filter_sorted, flatten, pprint
from devito.visitors import FindNodes, FindSymbols, Transformer


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._create_elemental_functions(state)

    @dle_pass
    def _avoid_denormals(self, state, **kwargs):
        """
        Introduce nodes in the Iteration/Expression tree that will expand to C
        macros telling the CPU to flush denormal numbers in hardware. Denormals
        are normally flushed when using SSE-based instruction sets, except when
        compiling shared objects.
        """
        return {'nodes': (Denormals(),) + state.nodes,
                'includes': ('xmmintrin.h', 'pmmintrin.h')}

    @dle_pass
    def _create_elemental_functions(self, state, **kwargs):
        """
        Extract :class:`Iteration` sub-trees and move them into :class:`Function`s.

        Currently, only tagged, elementizable Iteration objects are targeted.
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))

        functions = OrderedDict()
        processed = []
        for node in state.nodes:
            mapper = {}
            for tree in retrieve_iteration_tree(node, mode='superset'):
                # Search an elementizable sub-tree (if any)
                tagged = filter_iterations(tree, lambda i: i.tag is not None, 'asap')
                if not tagged:
                    continue
                root = tagged[0]
                if not root.is_Elementizable:
                    continue

                name = "f_%d" % len(functions)
                print(name)
                # Heuristic: create elemental functions only if more than
                # self.thresholds['elemental_functions'] operations are present
                expressions = FindNodes(Expression).visit(root)
                ops = estimate_cost([e.expr for e in expressions])
                if ops < self.thresholds['elemental'] and not root.is_Elementizable:
                    continue
                pprint(root)
                # Determine the arguments required by the elemental function
                in_scope = [i.dim for i in tree[tree.index(root):]]
                print(in_scope)
                required = FindSymbols(mode='free-symbols').visit(root)
                print(required)
                for i in FindSymbols('symbolics').visit(root):
                    required.extend(flatten(j.rtargs for j in i.indices))
                required = set([as_symbol(i) for i in required if i not in in_scope])
                print(required)
                # Add tensor arguments
                args = []
                seen = {e.output for e in expressions if e.is_scalar}
                for i in FindSymbols('symbolics').visit(root):
                    if i.is_SymbolicFunction:
                        handle = "(%s*) %s" % (c.dtype_to_ctype(i.dtype), i.name)
                    else:
                        handle = "%s_vec" % i.name
                    args.append((handle, i))

                    seen |= {as_symbol(i)}
                # Add scalar arguments
                handle = filter_sorted(required - seen, key=attrgetter('name'))
                print(handle)
                args.extend([(i.name, ScalarFunction(name=i.name, dtype=np.int32))
                             for i in handle])
                # Track info to transform the main tree
                call, parameters = zip(*args)
                mapper[view] = (List(header=noinline, body=FunCall(name, call)), [root])

                args = flatten([p.rtargs for p in parameters])
                print("****")
                # Produce the new function
                functions.append(Function(name, root, 'void', args, ('static',)))

            # Transform the main tree
            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed, 'elemental_functions': functions.values()}

    def _compiler_decoration(self, name, default=None):
        key = self.params['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)
