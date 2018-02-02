from __future__ import absolute_import

from collections import OrderedDict

from devito.ir import Cluster, ClusterGroup, groupby
from devito.dse.aliases import collect
from devito.dse.backends import BasicRewriter, dse_pass
from devito.symbolics import Eq, estimate_cost, xreplace_constrained, iq_timeinvariant
from devito.dse.manipulation import (common_subexprs_elimination, collect_nested,
                                     compact_temporaries)
from devito.types import Indexed, Scalar, Array


class AdvancedRewriter(BasicRewriter):

    def _pipeline(self, state):
        self._extract_time_invariants(state, costmodel=lambda e: e.is_Function)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _extract_time_invariants(self, cluster, template, with_cse=True,
                                 costmodel=None, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """

        # Extract time invariants
        make = lambda i: Scalar(name=template(i)).indexify()
        rule = iq_timeinvariant(cluster.trace)
        costmodel = costmodel or (lambda e: estimate_cost(e) > 0)
        processed, found = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        if with_cse:
            leaves = [i for i in processed if i not in found]

            # Search for common sub-expressions amongst them (and only them)
            make = lambda i: Scalar(name=template(i + len(found))).indexify()
            found = common_subexprs_elimination(found, make)

            # Some temporaries may be droppable at this point
            processed = compact_temporaries(found, leaves)

        return cluster.rebuild(processed)

    @dse_pass
    def _factorize(self, cluster, *args, **kwargs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              self.threshold, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in cluster.exprs:
            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle >= self.thresholds['min-cost-factorize']:
                handle_prev = handle
                cost_prev = estimate_cost(expr)
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_inter_stencil_redundancies(self, cluster, template, **kwargs):
        """
        Search for redundancies across the expressions and expose them
        to the later stages of the optimisation pipeline by introducing
        new temporaries of suitable rank.

        Two type of redundancies are sought:

            * Time-invariants, and
            * Across different space points

        Examples
        ========
        Let ``t`` be the time dimension, ``x, y, z`` the space dimensions. Then:

        1) temp = (a[x,y,z]+b[x,y,z])*c[t,x,y,z]
           >>>
           ti[x,y,z] = a[x,y,z] + b[x,y,z]
           temp = ti[x,y,z]*c[t,x,y,z]

        2) temp1 = 2.0*a[x,y,z]*b[x,y,z]
           temp2 = 3.0*a[x,y,z+1]*b[x,y,z+1]
           >>>
           ti[x,y,z] = a[x,y,z]*b[x,y,z]
           temp1 = 2.0*ti[x,y,z]
           temp2 = 3.0*ti[x,y,z+1]
        """
        if cluster.is_sparse:
            return cluster

        # For more information about "aliases", refer to collect.__doc__
        mapper, aliases = collect(cluster.exprs)

        # Redundancies will be stored in space-varying temporaries
        g = cluster.trace
        indices = g.space_indices
        time_invariants = {v.rhs: g.time_invariant(v) for v in g.values()}

        # Find the candidate expressions
        processed = []
        candidates = OrderedDict()
        for k, v in g.items():
            # Cost check (to keep the memory footprint under control)
            naliases = len(mapper.get(v.rhs, []))
            cost = estimate_cost(v, True)*naliases
            if cost >= self.thresholds['min-cost-alias'] and\
                    (naliases > 1 or time_invariants[v.rhs]):
                candidates[v.rhs] = k
            else:
                processed.append(Eq(k, v.rhs))

        # Create alias Clusters and all necessary substitution rules
        # for the new temporaries
        alias_clusters = ClusterGroup()
        rules = OrderedDict()
        for c, (origin, alias) in enumerate(aliases.items()):
            if all(i not in candidates for i in alias.aliased):
                continue
            # Retrieve the alias iteration space
            ispace = cluster.ispace.subtract(alias.anti_stencil.boxify().negate())
            # Build a symbolic function for the alias
            shape = tuple(ispace[i].extent for i in indices)
            function = Array(name=template(c), shape=shape, dimensions=indices)
            # Build alias Cluster
            access = tuple(i - ispace[i].lower for i in indices)
            expression = Eq(Indexed(function.indexed, *access), origin)
            if all(time_invariants[i] for i in alias.aliased):
                ispace = ispace.drop([i.dim for i in ispace.intervals if i.dim.is_Time])
            alias_clusters.append(Cluster([expression], ispace))
            # Add substitution rules
            for aliased, distance in alias.with_distance:
                access = tuple(i - ispace[i].lower + j for i, j in distance.items())
                temporary = Indexed(function.indexed, *tuple(access))
                rules[candidates[aliased]] = temporary
                rules[aliased] = temporary
        alias_clusters = groupby(alias_clusters).finalize()
        alias_clusters.sort(key=lambda i: i.is_dense)

        # Switch temporaries in the expression trees
        processed = [e.xreplace(rules) for e in processed]

        return alias_clusters + [cluster.rebuild(processed)]
