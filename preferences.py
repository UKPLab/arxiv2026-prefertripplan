"""
preferences.py — Full preference type hierarchy
================================================

Each Preference subclass implements two interfaces:
  to_z3(instances, solver)          → used during plan GENERATION
  evaluate(plan)  → CheckResult     → used during plan EVALUATION

Preference types
----------------
  AtomicPreference          single attribute constraint
  CompositePreference       AND / OR / NOT over sub-preferences
  ConditionalPreference     IF cond THEN p1 [ELSE p2]
  LexicographicPreference   ordered priority list (p1 > p2 > p3 …)
  NumericPreference        prefer min/max of a numeric attribute
  TemporalPreference   9 temporal modality operations based on PDDL 
                            temporal constraints between entity A and entity B 
                            in plan sequence
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np

# Z3 imports (lazy-safe: only fail at to_z3 call time if z3 missing)
try:
    from z3 import (
        And, Or, Not, Implies, BoolRef,
        EnumSort, IntSort, RealSort,
        Int, Real, Const, Bool,
        Optimize, sat,
        IntVal, RealVal,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintType(str, Enum):
    COMMONSENSE = "commonsense"
    HARD        = "hard"
    PREFERENCE  = "preference"


@dataclass
class CheckResult:
    """
    ``details`` now carries STRUCTURED provenance instead of a bare
    string.  Every ``details`` dict is guaranteed to have:

      * ``kind``    (str)  paradigm-and-branch discriminator, e.g.
                     ``"atomic"``, ``"composite"``, ``"lex"``,
                     ``"numeric"``, ``"scoped"``, ``"compensatory"``,
                     ``"temporal.group"``, ``"temporal.within"``,
                     ``"temporal.timed"``, ``"temporal.pair"``,
                     ``"temporal.aggregate"``, or a ``.<sub>`` variant
                     (e.g. ``"atomic.no_entities"``, ``"scoped.no_match"``)
                     for early-exit branches.
      * ``message`` (str)  human-readable rendering (the same text that
                     used to be the entire ``details`` field).  Keeps
                     ``__repr__`` / debug prints legible without any
                     caller change.

    Every paradigm-specific field beyond those two is documented at the
    corresponding construction site inside each ``evaluate()``.  The
    field is always a dict; downstream analysis code should pivot on
    ``details["kind"]`` and access the paradigm-specific keys directly.
    """
    name:             str
    constraint_type:  ConstraintType
    passed:           bool
    score:            float        # 0.0–1.0 (hard: binary; soft: partial ok)
    trivial:          bool = False
    details:          dict = field(default_factory=dict)
    sub_results:      list["CheckResult"] = field(default_factory=list)

    def __repr__(self):
        mark = "✓" if self.passed else "✗"
        msg  = self.details.get("message", "") if isinstance(self.details, dict) \
               else str(self.details)
        return f"[{mark}] {self.name}  score={self.score:.2f}  — {msg}"


def _details(kind: str, message: str, **fields: Any) -> dict[str, Any]:
    """Assemble a structured ``CheckResult.details`` dict.  Every site
    that constructs a ``CheckResult`` funnels through this so ``kind``
    and ``message`` are present on every record and downstream analysis
    can pivot on ``kind`` without probing every construction site."""
    return {"kind": kind, "message": message, **fields}


# ─────────────────────────────────────────────────────────────────────────────
# Domain layer (same as before, slimmed)
# ─────────────────────────────────────────────────────────────────────────────

class Domain:
    def encode(self, value): raise NotImplementedError
    def make_z3_var(self, name): raise NotImplementedError
    def valid_range_constraint(self, var): raise NotImplementedError


# Global registry: tuple(values) → (sort, consts_dict)
# Prevents "enumeration sort name is already declared" when the same value
# list is used across multiple CategoricalDomain instances in the same process
# (e.g. build_entity_specs_for_city called for Paris then Tokyo both declare
# CategoricalDomain(["first", "last"]) which would otherwise collide in Z3).
_ENUM_SORT_REGISTRY: dict = {}


@dataclass
class CategoricalDomain(Domain):
    values: list[str]
    _sort: Any = field(init=False, default=None)
    _consts: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if not Z3_AVAILABLE:
            return
        key = tuple(self.values)
        if key in _ENUM_SORT_REGISTRY:
            # Reuse existing Z3 sort — redeclaring the same name raises Z3Exception
            self._sort, self._consts = _ENUM_SORT_REGISTRY[key]
        else:
            import hashlib
            h         = hashlib.md5("|".join(self.values).encode()).hexdigest()[:8]
            sort_name = "Cat_" + h
            sort, consts = EnumSort(sort_name, self.values)
            self._sort   = sort
            self._consts = dict(zip(self.values, consts))
            _ENUM_SORT_REGISTRY[key] = (self._sort, self._consts)

    def encode(self, v):
        return self._consts[v]

    def make_z3_var(self, name):
        return Const(name, self._sort)

    def valid_range_constraint(self, var):
        return Or([var == c for c in self._consts.values()])

    def py_check(self, actual, op, value):
        # Scalar categorical value.  For list-valued attributes the
        # authoritative eval is ``AtomicPreference._check_one``; this
        # helper is a domain-level scalar approximation kept for
        # consistency, so a caller that hands a scalar ``actual`` and
        # a set ``value`` gets a sensible answer.
        if op == "in":          return str(actual) in value
        if op == "not_in":      return str(actual) not in value
        if op == "contains_all":
            # `value` must be a subset of `{actual}` (a singleton).
            # Only true when `value` is either empty or {actual}.
            return all(str(v) == str(actual) for v in value)
        if op == "==":          return str(actual) == str(value)
        if op == "!=":          return str(actual) != str(value)
        raise ValueError(f"Unsupported op '{op}' on CategoricalDomain")


@dataclass
class NumericDomain(Domain):
    lo: float
    hi: float
    is_int: bool = True

    def encode(self, v):
        return IntVal(int(v)) if self.is_int else RealVal(str(round(v, 2)))

    def make_z3_var(self, name):
        return Int(name) if self.is_int else Real(name)

    def valid_range_constraint(self, var):
        lo = IntVal(int(self.lo)) if self.is_int else RealVal(str(self.lo))
        hi = IntVal(int(self.hi)) if self.is_int else RealVal(str(self.hi))
        return And(var >= lo, var <= hi)

    def py_check(self, actual, op, value):
        try:
            a = float(actual)
        except (TypeError, ValueError):
            return False
        ops = {"==": a == value, "!=": a != value,
               "<":  a <  value, "<=": a <= value,
               ">":  a >  value, ">=": a >= value,
               "range": value[0] <= a <= value[1]}
        return ops[op]


# ─────────────────────────────────────────────────────────────────────────────
# Base Preference
# ─────────────────────────────────────────────────────────────────────────────

class Preference(ABC):
    """Abstract base.  weight=None → hard constraint."""

    weight: float | None = None   # subclasses set in __init__
    name:   str          = ""

    # ── generation side ──────────────────────────────────────────────────────
    def to_z3(self, instances: dict, solver) -> None:
        """Compile self into solver (add / add_soft)."""
        raise NotImplementedError(f"{type(self).__name__}.to_z3 not implemented")

    def _add(self, solver, expr):
        """Route to add() or add_soft() based on weight."""
        if self.weight is None:
            solver.add(expr)
        else:
            solver.add_soft(expr, weight=self.weight)

    # ── evaluation side ───────────────────────────────────────────────────────
    @abstractmethod
    def evaluate(self, plan: list) -> CheckResult:
        """Evaluate self against a concrete plan dict."""

    # ── convenience ──────────────────────────────────────────────────────────
    def __and__(self, other):  return CompositePreference("AND", [self, other])
    def __or__(self,  other):  return CompositePreference("OR",  [self, other])
    def __invert__(self):      return CompositePreference("NOT", [self])


# ─────────────────────────────────────────────────────────────────────────────
# 1. AtomicPreference
# ─────────────────────────────────────────────────────────────────────────────

class AtomicPreference(Preference):
    """
    Constrains one attribute of one entity type.

    Parameters
    ----------
    entity_type : str          e.g. "Restaurant"
    attribute   : str          e.g. "rating"
    op          : str          One of:
                                 "==" | "!=" | "<" | "<=" | ">" | ">="
                                     Scalar comparisons.
                                 "range"
                                     ``value`` is a 2-tuple ``(lo, hi)``;
                                     matches iff the entity's value is in
                                     the inclusive interval.
                                 "in" | "not_in"
                                     Any-overlap / no-overlap set ops
                                     against ``value`` (a list of strings).
                                     ``in``      → ``entity_vals ∩ value ≠ ∅``
                                     ``not_in``  → ``entity_vals ∩ value = ∅``
                                     Use for e.g. cuisines where a single
                                     match is enough.
                                 "contains_all"
                                     Strict superset check: every element
                                     of ``value`` must be present in the
                                     entity's list-valued attribute.
                                     ``value ⊆ entity_vals``.  Use for
                                     ``Accommodation.house_rules`` where
                                     each listed rule must actually apply
                                     on the accommodation -- one missing
                                     rule fails the check.  For list-
                                     valued attributes only; on a scalar
                                     attribute this is only satisfied
                                     when ``value`` is a singleton
                                     equalling the entity's value.
    value       : any          list for "in" / "not_in" / "contains_all";
                               (lo, hi) tuple for "range";
                               scalar otherwise.
    scope       : "any"|"all"  "any" → at least one entity passes;
                               "all" → every entity must.
    weight      : float|None   None = hard
    name        : str          human label
    """

    def __init__(self, entity_type: str, attribute: str, op: str, value,
                 scope: str = "any", weight: float | None = None, name: str = ""):
        self.entity_type = entity_type
        self.attribute   = attribute
        self.op          = op
        self.value       = value
        self.scope       = scope          # "any" or "all"
        self.weight      = weight
        self.name        = name or f"Atomic:{entity_type}.{attribute} {op} {value} [{scope}]"

    # ── shorthand factory ────────────────────────────────────────────────────
    @classmethod
    def parse(cls, entity_type: str, attribute: str, spec,
              scope="any", weight=None, name="") -> "AtomicPreference":
        """Concise-syntax factory.  Maps a lightweight ``spec`` value to
        an :class:`AtomicPreference` with the ``op`` inferred from
        ``spec``'s Python type / shape:

        ============================================  =========  =====================
        ``spec``                                       inferred    example
                                                       ``op``
        ============================================  =========  =====================
        str prefixed by ``>=`` / ``<=`` / ``!=`` /
          ``>`` / ``<`` / ``==``                       that op    ``">=4.5"``
        list (any length) / numpy 1-D array            ``"in"``   ``["Italian","Thai"]``
        2-tuple ``(lo, hi)``                           ``"range"``  ``(100, 200)``
        anything else (scalar)                         ``"=="``   ``"Vegetarian"``
        ============================================  =========  =====================

        Design choice + repercussions
        -----------------------------
        These four inference paths are the ONLY ones covered here.
        Ops that don't have an unambiguous shorthand -- specifically
        ``"not_in"``, ``"contains_all"``, and ``"range"`` with a
        non-2-tuple form -- are NOT reachable through this factory
        because:

          * ``not_in``: a list ``spec`` is already claimed by ``"in"``
            (any-overlap).  There's no natural syntactic marker that
            distinguishes "membership" from "non-membership" without
            re-encoding it into a wrapper / sentinel value.
          * ``contains_all``: a list ``spec`` looks identical to the
            ``"in"`` case -- and the two semantics are exactly what
            we want to keep visibly distinct in the preference JSON.
            Silently overloading ``list -> in`` with an implicit mode
            switch would defeat the point of adding the new op.
          * non-2-tuple ``range``: keeping the tuple length as the
            disambiguator between ``"range"`` and ``"=="`` on a
            general tuple keeps the type→op mapping unambiguous.

        Callers that need any of the above MUST construct the pref
        through the full constructor:

            AtomicPreference("Accommodation", "house_rules",
                             "contains_all", ["No smoking", "No pets"],
                             scope="all")

        See the class docstring for the full op vocabulary."""
        if isinstance(spec, str) and re.match(r"^(>=|<=|!=|>|<|==)", spec):
            m   = re.match(r"^(>=|<=|!=|>|<|==)\s*(.+)$", spec)
            op  = m.group(1)
            raw = m.group(2)
            try:   val = int(raw) if "." not in raw else float(raw)
            except ValueError: val = raw
            return cls(entity_type, attribute, op, val, scope, weight, name)
        if isinstance(spec, (list, np.ndarray)):
            return cls(entity_type, attribute, "in", list(spec), scope, weight, name)
        if isinstance(spec, tuple) and len(spec) == 2:
            return cls(entity_type, attribute, "range", spec, scope, weight, name)
        return cls(entity_type, attribute, "==", spec, scope, weight, name)

    # ── evaluation ───────────────────────────────────────────────────────────
    def evaluate(self, plan: list) -> CheckResult:
        entities = _collect(plan, self.entity_type)
        if not entities:
            return CheckResult(
                self.name, _ctype(self.weight), False, 0.0, False,
                _details(
                    "atomic.no_entities",
                    f"No {self.entity_type} found in plan",
                    entity_type     = self.entity_type,
                    attribute       = self.attribute,
                    op              = self.op,
                    value           = self.value,
                    scope           = self.scope,
                    n_entities      = 0,
                    passed_entities = 0,
                ),
            )

        results = [self._check_one(e) for e in entities]
        passed_count = sum(results)

        if self.scope == "any":
            passed = any(results)
            score  = float(passed)
        else:  # "all"
            passed = all(results)
            score  = passed_count / len(results)
        # score  = passed_count / len(results)

        return CheckResult(
            self.name, _ctype(self.weight), passed, score, False,
            _details(
                "atomic",
                f"{passed_count}/{len(results)} {self.entity_type}s satisfy "
                f"{self.attribute} {self.op} {self.value}",
                entity_type     = self.entity_type,
                attribute       = self.attribute,
                op              = self.op,
                value           = self.value,
                scope           = self.scope,
                n_entities      = len(results),
                passed_entities = int(passed_count)/len(results),
            ),
        )

    def _check_one(self, entity: dict) -> bool:
        raw = entity.get(self.attribute)
        if raw is None:
            return False
        op, val = self.op, self.value

        # An entity's attribute may be a list (e.g. cuisine =
        # ["French", "Mexican"]) or a scalar (e.g. rating = 4.5).
        # Normalise to a list so set-membership ops (in / not_in /
        # contains_all) have a uniform view; scalar-comparison ops
        # (== / != / < / <= / > / >= / range) still read the first
        # element via ``entity_vals[0]``.
        entity_vals     = raw if isinstance(raw, (list, tuple)) else [raw]
        entity_val_strs = [str(ev) for ev in entity_vals]

        if op == "in":
            # Any-overlap: at least one entity value is in the preference set.
            # Suitable for cuisines where a single match is enough.
            val_strs = [str(v) for v in val]
            return any(ev in val_strs for ev in entity_val_strs)
        if op == "not_in":
            # No-overlap: no entity value appears in the preference set.
            # Suitable for "avoid these rules / cuisines / ..." templates.
            val_strs = [str(v) for v in val]
            return all(ev not in val_strs for ev in entity_val_strs)
        if op == "contains_all":
            # Superset: every value in the preference list is present in
            # the entity's list-valued attribute.  Suitable for
            # accommodation house_rules where the preference names rules
            # the accommodation must abide by -- one missing = fail.
            val_strs = [str(v) for v in val]
            return all(v in entity_val_strs for v in val_strs)
        if op == "range":
            try:
                return any(val[0] <= float(ev) <= val[1]
                           for ev in entity_val_strs)
            except (TypeError, ValueError):
                return False
        # Numeric / string comparison — use the first scalar entity value.
        try:
            ev0 = float(entity_vals[0])
            ops = {"==": ev0==val, "!=": ev0!=val,
                   "<":  ev0<val,  "<=": ev0<=val,
                   ">":  ev0>val,  ">=": ev0>=val}
            return ops.get(op, False)
        except (TypeError, ValueError):
            s = str(entity_vals[0])
            if op == "==": return s == str(val) or str(val) in entity_val_strs
            if op == "!=": return s != str(val) and str(val) not in entity_val_strs
            return False

    # ── Z3 generation ────────────────────────────────────────────────────────
    def to_z3(self, instances: dict, solver) -> None:
        entity_instances = instances.get(self.entity_type, [])
        exprs = [self._expr_for(inst) for inst in entity_instances]
        if not exprs:
            return
        combined = Or(exprs) if self.scope == "any" else And(exprs)
        self._add(solver, combined)

    def _expr_for(self, inst) -> BoolRef:
        from z3 import BoolVal as _BoolVal
        attr_spec = inst.spec.attributes.get(self.attribute)
        var = inst.vars[self.attribute]
        op, val = self.op, self.value
        is_categorical = isinstance(attr_spec.domain, CategoricalDomain) if attr_spec else False

        if op == "in":
            # val may be a scalar or list; preference value may also be a list
            val_list = val if isinstance(val, (list, tuple)) else [val]
            exprs = []
            for v in val_list:
                try:
                    exprs.append(var == attr_spec.domain.encode(v))
                except (KeyError, ValueError):
                    pass
            return Or(exprs) if exprs else _BoolVal(False)
        if op == "not_in":
            val_list = val if isinstance(val, (list, tuple)) else [val]
            exprs = []
            for v in val_list:
                try:
                    exprs.append(var != attr_spec.domain.encode(v))
                except (KeyError, ValueError):
                    pass
            return And(exprs) if exprs else _BoolVal(True)
        if op == "contains_all":
            # ``contains_all`` requires every value in ``val`` to be
            # present in the entity's attribute.  Z3's scalar-var model
            # for a single attribute can't natively represent a set-
            # valued instance, so the strict subset check is only
            # cleanly encodable when ``val`` is a singleton (a single
            # required value the var must equal).  For multi-value
            # ``val`` on a list-valued real-world attribute (e.g.
            # accommodation house_rules), authoritative enforcement
            # happens in ``_check_one`` at evaluation time; here we emit
            # a conservative ``And(var == encode(v) for v in val)``
            # which is only satisfiable for a singleton val and hence
            # simply doesn't constrain the solver for multi-value
            # cases (a subsequent post-hoc filter is expected to enforce
            # membership on real data).
            val_list = val if isinstance(val, (list, tuple)) else [val]
            exprs = []
            for v in val_list:
                try:
                    exprs.append(var == attr_spec.domain.encode(v))
                except (KeyError, ValueError):
                    pass
            return And(exprs) if exprs else _BoolVal(False)
        if op == "range":
            # Only valid for numeric domains
            if is_categorical:
                return _BoolVal(True)   # can't order categoricals; skip
            return And(var >= val[0], var <= val[1])
        # Scalar comparison ops
        if is_categorical:
            # Only == and != are valid on EnumSort (DatatypeRef)
            try:
                enc = attr_spec.domain.encode(val)
            except (KeyError, ValueError):
                return _BoolVal(op == "!=")
            if op == "==": return var == enc
            if op == "!=": return var != enc
            return _BoolVal(True)   # <, <=, >, >= not supported on categorical
        try:
            enc = attr_spec.domain.encode(val)
        except (KeyError, ValueError):
            return _BoolVal(False)
        ops_map = {"==": var==enc, "!=": var!=enc,
                   "<":  var<enc,  "<=": var<=enc,
                   ">":  var>enc,  ">=": var>=enc}
        return ops_map.get(op, _BoolVal(True))


# ─────────────────────────────────────────────────────────────────────────────
# 2. CompositePreference  (AND / OR / NOT)
# ─────────────────────────────────────────────────────────────────────────────

class CompositePreference(Preference):
    """
    Boolean composition of sub-preferences.

    op    : "AND" | "OR" | "NOT"
    children : list of Preference  (NOT takes exactly one)
    """

    def __init__(self, op: str, children: list[Preference],
                 weight: float | None = None, name: str = ""):
        assert op in ("AND", "OR", "NOT")
        # assert op in ("AND", "OR")
        if op == "NOT":
            assert len(children) == 1
        self.op       = op
        self.children = children
        self.weight   = weight
        self.name     = name or f"Composite:({op} {'; '.join(c.name for c in children)})"

    def evaluate(self, plan: list) -> CheckResult:
        sub = [c.evaluate(plan) for c in self.children]
        if self.op == "AND":
            passed = all(r.passed for r in sub)
            score  = sum(r.score for r in sub) / len(sub)
        elif self.op == "OR":
        # else:  # OR
            passed = any(r.passed for r in sub)
            score  = max(r.score for r in sub)
        else:  # NOT
            passed = not sub[0].passed
            score  = 1.0 - sub[0].score
        n_passed_children = sum(1 for r in sub if r.passed)
        return CheckResult(
            self.name, _ctype(self.weight), passed, score, False,
            _details(
                "composite",
                f"{self.op} of {len(sub)} sub-prefs",
                op              = self.op,
                n_children      = len(sub),
                passed_children = (n_passed_children / len(sub)) if sub else 0,
            ),
            sub,
        )

    def to_z3(self, instances: dict, solver) -> None:
        from z3 import Bool as Z3Bool
        # collect sub Z3 bool vars by adding to a temp solver
        child_bools = []
        for child in self.children:
            b = Z3Bool(f"_cp_{id(child)}")
            # Encode: b ↔ child_expr  (via implication both ways)
            # Simpler approach: inline the child directly
            child_bools.append(_child_expr(child, instances))

        if self.op == "AND":
            expr = And(child_bools)
        elif self.op == "OR":
            expr = Or(child_bools)
        else:
            expr = Not(child_bools[0])
        self._add(solver, expr)


def _child_expr(pref: Preference, instances: dict):
    """Return a raw Z3 expression for a preference (for composition use)."""
    from z3 import BoolVal
    if isinstance(pref, AtomicPreference):
        entity_instances = instances.get(pref.entity_type, [])
        exprs = [pref._expr_for(inst) for inst in entity_instances]
        if not exprs:
            return BoolVal(True)
        return Or(exprs) if pref.scope == "any" else And(exprs)
    if isinstance(pref, CompositePreference):
        child_exprs = [_child_expr(c, instances) for c in pref.children]
        if pref.op == "AND":  return And(child_exprs)
        if pref.op == "OR":   return Or(child_exprs)
        return Not(child_exprs[0])
    # Fallback: add directly and return True sentinel
    pref.to_z3(instances, _NullSolver())
    from z3 import BoolVal
    return BoolVal(True)


class _NullSolver:
    """Absorbs to_z3 calls during composition building."""
    def add(self, *a): pass
    def add_soft(self, *a, **k): pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. ConditionalPreference
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalPreference(Preference):
    """
    IF condition THEN then_pref [ELSE else_pref]

    condition : Preference (used as a boolean test)
    Condition evaluates existence in global plan and accordingly then_pref or 
    else_pref existence is checked in global plan. For example, if a plan 
    contains event category A (say hiking), it is also preferred to have event 
    category B (say spa). These may be due to conditional attraction satisfaction 
    for multiple users. Thus, then_pref or else_pref are NOT to be limited to the 
    condition scope. Such cases are defined by ScopedPreferences where the 
    condition confines the scope as well.
    """

    def __init__(self, condition: Preference,
                 then_pref: Preference,
                 else_pref: Preference | None = None,
                 weight: float | None = None,
                 name: str = ""):
        self.condition = condition
        self.then_pref = then_pref
        self.else_pref = else_pref
        self.weight    = weight
        self.name      = name or f"Conditional:IF {condition.name} THEN {then_pref.name}"
        if else_pref:
            self.name += f" ELSE {else_pref.name}"

    def evaluate(self, plan: list) -> CheckResult:
        """
        Original whole-plan evaluation — used for same-entity conditionals."""
        cond_result = self.condition.evaluate(plan)
        if cond_result.passed:
            active, branch = self.then_pref, "then"
            if active is None:
                return CheckResult(
                    self.name, _ctype(self.weight), True, 1.0, True,
                    _details(
                        "conditional.trivial",
                        "Condition true but then_pref=None — trivially satisfied",
                        branch          = "then",
                        cond_passed     = True,
                        active_branch_present = False,
                    ),
                    [cond_result],
                )
        else:
            if self.else_pref is None:
                return CheckResult(
                    self.name, _ctype(self.weight), True, 1.0, True,
                    _details(
                        "conditional.trivial",
                        "Condition false and no else branch — trivially satisfied",
                        branch          = "else",
                        cond_passed     = False,
                        active_branch_present = False,
                    ),
                    [cond_result],
                )
            active, branch = self.else_pref, "else"
        active_result = active.evaluate(plan)
        return CheckResult(
            self.name, _ctype(self.weight),
            active_result.passed, active_result.score, False,
            _details(
                "conditional",
                f"Condition {'true' if branch=='then' else 'false'} → "
                f"evaluating {branch} branch",
                branch        = branch,
                cond_passed   = bool(cond_result.passed),
                active_passed = bool(active_result.passed),
                active_score  = float(active_result.score),
            ),
            [cond_result, active_result],
        )

    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporintg generation same as evaluation logic above
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 4. LexicographicPreference
# ─────────────────────────────────────────────────────────────────────────────

class LexicographicPreference(Preference):
    """
    Ordered list of preferences: p1 > p2 > ... > pn.

    Semantics
    ---------
    Any improvement at tier i strictly outweighs maximally satisfying
    all lower tiers i+1 ... n combined, encoded as a single scalar in
    [0, 1] via big-M weighting.

    Only independent evaluation is supported: each tier evaluates its
    own entity pool without interference, making this valid for:
      - Different entity types  (Accommodation.rating > Restaurant.rating)
      - Different attributes    (Restaurant.price > Restaurant.rating)
      - Same attributes (Restaurant.cuisine1 > Restaurant.cuisine2)
      - Mixed / composite tiers (AtomicPreference, nested Lex, etc.)
    Independence" wording is about score independence, not entity-pool independence. 
    Although the more overlap between entity bool has the lesser practical function
    lexicographic tiers does.

    Big-M correctness guarantee
    ---------------------------
    Score resolution at tier i is 1/N_i (one entity changing state).
    For any improvement at tier i to dominate all lower tiers combined:

        w_i / N_i  >  Σ_{j>i} w_j

    With w_i = M^(n-1-i) this reduces to M > N_i for every i, giving:

        M = max(N_i) + 1

    Parameters
    ----------
    preferences  : list[Preference]
        Sub-preferences in descending priority order.
    weight       : float | None
        Overall importance; None = hard constraint.
    name         : str
        Human-readable label.
    max_entities : int | None
        Fixed upper bound on entity count used to compute M.
        - None  → M derived dynamically per plan.
                  Safe only when all compared plans have identical
                  entity counts per type.
        - int   → M = max(max_entities, dynamic_max) + 1.
                  Set this to the largest entity count any valid plan
                  can have to guarantee scalar comparability across
                  plans of different lengths.
    """

    def __init__(
        self,
        preferences:  list[Preference],
        weight:       float | None = 1.0,
        name:         str = "",
        max_entities: int | None = None,
    ):
        self.preferences  = preferences
        self.weight       = weight
        self.name         = name or "Lexicographic:(" + "; ".join(p.name for p in preferences) + ")"
        self.max_entities = max_entities

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, plan: list) -> CheckResult:
        if not self.preferences:
            return CheckResult(
                self.name, ConstraintType.PREFERENCE, True, 1.0, False,
                _details("lex.empty", "No sub-preferences defined",
                         n_tiers=0),
            )

        sub_results:   list[CheckResult] = []
        tier_scores:   list[float]       = []
        entity_counts: list[int]         = []

        for p in self.preferences:
            result = p.evaluate(plan)
            sub_results.append(result)
            tier_scores.append(result.score)
            entity_counts.append(self._entity_count(plan, p))

        scalar, M = self._big_m_scalar(tier_scores, entity_counts)

        passed = sub_results[0].passed
        detail_msg = (
            f"M={M}  "
            + "  ".join(
                f"P{i+1}[{r.name}]={s:.3f}({'✓' if r.passed else '✗'})"
                for i, (r, s) in enumerate(zip(sub_results, tier_scores))
            )
        )
        tiers = [
            {
                "index":  i,
                "name":   r.name,
                "score":  float(s),
                "passed": bool(r.passed),
                "entity_count": int(entity_counts[i]),
            }
            for i, (r, s) in enumerate(zip(sub_results, tier_scores))
        ]
        n_passed_tiers = sum(1 for r in sub_results if r.passed)
        return CheckResult(
            self.name, ConstraintType.PREFERENCE,
            passed, scalar, False,
            _details(
                "lex",
                detail_msg,
                M            = int(M),
                n_tiers      = len(sub_results),
                passed_tiers = (n_passed_tiers / len(sub_results))
                                if sub_results else 0,
                tiers        = tiers,
                scalar       = float(scalar),
            ),
            sub_results,
        )

    # ── big-M scalar ─────────────────────────────────────────────────────────

    def _big_m_scalar(
        self,
        tier_scores:   list[float],
        entity_counts: list[int],
    ) -> tuple[float, int]:
        """
        Encode lex-ordered scores into a single normalised scalar in [0, 1].

        M = max(entity_counts, max_entities) + 1 ensures any single-step
        improvement at tier i (score gain of 1/N_i) strictly outweighs
        maximal satisfaction of all lower tiers combined.
        """
        n       = len(tier_scores)
        N_max   = max(entity_counts) if entity_counts else 1
        M       = max(N_max, self.max_entities or 0) + 1

        raw     = sum(M ** (n - 1 - i) * s for i, s in enumerate(tier_scores))
        max_raw = (M ** n - 1) / (M - 1)   # geometric series: M^(n-1) + ... + 1
        scalar  = raw / max_raw

        return scalar, M

    # ── helpers ──────────────────────────────────────────────────────────────

    def _entity_count(self, plan: list, p: Preference) -> int:
        """
        Entity count used to determine the finest score resolution at this tier.

        AtomicPreference : actual entity count from the plan  (score step = 1/N)
        anything else    : 1, since composite scores are already in [0, 1]
                           with no finer resolution knowable externally.
                           Using 1 is conservative — never underestimates M.
        """
        if isinstance(p, AtomicPreference):
            return max(len(_collect(plan, p.entity_type)), 1)
        # TODO, right now restrict individual preferences to Atomic only. 
        # Check and include composites or complex preferences later.
        # E.g. ScopedPreferences inside Lexicographic preferences can lead to 
        # underestimating entitiy count and hence M. Treat the below as just a 
        # placeholder.
        return 1 


    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporintg generation same as evaluation logic above
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 5. NumericPreference  (prefer more/less of a numeric attribute)
# ─────────────────────────────────────────────────────────────────────────────

class NumericPreference(Preference):
    """
    Express 'the lower/higher the better' for a numeric attribute.

    direction : "min" → prefer smaller values (e.g. price)
                "max" → prefer larger values  (e.g. rating)
    threshold : list-of-5 quartile anchors ``[Q0, Q1, Q2, Q3, Q4]``
                sampling the query's own pool distribution.  The
                augmenter's ``resolve_numeric_dict`` emits this shape
                and downstream tooling (``patch_numeric_threshold.py``)
                migrates any legacy 2-tuple records to it before
                evaluation.

                For backward-compatibility, a 2-tuple ``[lo, hi]``
                passed at construction time is auto-expanded to a
                5-tuple by equal interpolation (``d = (hi-lo)/4``):
                ``[lo, lo+d, lo+2d, lo+3d, hi]``.  This lets
                external callers (e.g. hand-written test fixtures,
                legacy record loaders) evaluate cleanly without
                pre-processing.
    aggregation: "avg"|"sum"|"min"|"max" across multiple entities

    Passing semantics
    -----------------
    With a full 5-tuple threshold:
        direction="min" : passed iff  agg_val <= Q1
                          (aggregated value lands in the best 25%
                           of the observed pool distribution)
        direction="max" : passed iff  agg_val >= Q3
                          (aggregated value lands in the top 25%)

    Score = piecewise-linear "quantile position achieved," giving
      * 1.0 at the best-end anchor (Q0 for min, Q4 for max),
      * 0.75 at the pass boundary (Q1 for min, Q3 for max),
      * 0.5 at the median (Q2),
      * 0.25 at the other pass boundary (Q3 for min, Q1 for max),
      * 0.0 at the worst-end anchor.
    Linear interpolation between adjacent anchors.
    """

    def __init__(self, entity_type: str, attribute: str,
                 direction: str = "min",
                 threshold: list | None = None,
                 aggregation: str = "avg",
                 weight: float | None = 1.0,
                 name: str = ""):
        assert direction in ("min", "max")
        assert aggregation in ("avg", "sum", "min", "max")
        self.entity_type  = entity_type
        self.attribute    = attribute
        self.direction    = direction
        # Normalise threshold shape at construction time so downstream
        # code sees only the canonical 5-tuple (or None).  A 2-tuple
        # ``[lo, hi]`` is auto-expanded by equal interpolation.
        self.threshold    = self._expand_threshold(threshold)
        self.aggregation  = aggregation
        self.weight       = weight
        self.name         = name or f"Numeric:{entity_type}.{attribute} prefer-{direction} of aggregate-{aggregation}"

    @staticmethod
    def _expand_threshold(threshold: list | None) -> list | None:
        """Normalise a threshold argument to the canonical 5-tuple shape.

        ``None`` / ``[]`` → ``None`` (soft-optimisation, no pass boundary).
        5-tuple ``[Q0, Q1, Q2, Q3, Q4]`` → returned unchanged.
        2-tuple ``[lo, hi]`` → expanded to ``[lo, lo+d, lo+2d, lo+3d, hi]``
                with ``d = (hi-lo)/4`` (equal interpolation of the middle
                three anchors between the observed endpoints).
        Any other shape → returned as-is (``evaluate`` reports a
        schema-mismatch CheckResult; caller decides how to react).
        """
        if not threshold:
            return None
        if not isinstance(threshold, (list, tuple)):
            return threshold
        if len(threshold) == 5:
            return [float(x) for x in threshold]
        if len(threshold) == 2:
            lo, hi = float(threshold[0]), float(threshold[1])
            if hi < lo:
                hi = lo
            d = (hi - lo) / 4.0
            return [lo, lo + d, lo + 2 * d, lo + 3 * d, hi]
        return list(threshold)

    @staticmethod
    def _quantile_position(val: float, anchors: list[float]) -> float:
        """Piecewise-linear position of ``val`` along a sorted list of
        anchors, returned in ``[0, 1]``.  For an evenly spaced 5-anchor
        list the returned value is exactly 0, 0.25, 0.5, 0.75, 1 at
        each anchor; interpolated linearly in between.  Values below
        anchors[0] clip to 0.0; above anchors[-1] clip to 1.0."""
        n = len(anchors)
        if n == 0:                     return 0.0
        if val <= anchors[0]:          return 0.0
        if val >= anchors[-1]:         return 1.0
        for i in range(n - 1):
            lo, hi = anchors[i], anchors[i + 1]
            if lo <= val <= hi:
                local = (val - lo) / (hi - lo) if hi > lo else 0.0
                return (i + local) / (n - 1)
        return 1.0

    def evaluate(self, plan: list) -> CheckResult:
        entities = _collect(plan, self.entity_type)
        vals = []
        for e in entities:
            v = e.get(self.attribute)
            if v is not None:
                try: vals.append(float(v))
                except: pass
        if not vals:
            return CheckResult(
                self.name, ConstraintType.PREFERENCE, False, 0.0, False,
                _details(
                    "numeric.no_values",
                    f"No {self.entity_type}.{self.attribute} found",
                    entity_type = self.entity_type,
                    attribute   = self.attribute,
                    aggregation = self.aggregation,
                    direction   = self.direction,
                ),
            )

        agg_fns = {"avg": lambda x: sum(x)/len(x),
                   "sum": sum, "min": min, "max": max}
        agg_val = agg_fns[self.aggregation](vals)

        # Defensive re-normalise: __init__ already ran _expand_threshold,
        # but a caller may have mutated self.threshold post-construction
        # (test fixtures, ad-hoc scripting).  Re-expand so a 2-tuple
        # assigned after construction still evaluates cleanly.
        th = self._expand_threshold(self.threshold)
        if not th:
            # No threshold declared → treat as purely-optimisation soft
            # preference: always "satisfies."
            return CheckResult(
                self.name, ConstraintType.PREFERENCE, True, 1.0, False,
                _details(
                    "numeric.no_threshold",
                    f"{self.aggregation}({self.attribute})={agg_val:.2f}  "
                    f"direction={self.direction}  threshold=None (soft-optimisation)",
                    entity_type = self.entity_type,
                    attribute   = self.attribute,
                    aggregation = self.aggregation,
                    direction   = self.direction,
                    agg_value   = float(agg_val),
                    threshold   = None,
                    n_values    = len(vals),
                ),
            )

        # Only shape accepted downstream: 5-tuple ``[Q0, Q1, Q2, Q3, Q4]``.
        # _expand_threshold normalises 2-tuples but returns other shapes
        # verbatim; anything not length-5 is a schema violation.  Do not
        # crash, but report ``passed=False, score=0`` and label the record
        # so the caller sees the mismatch in the summary.
        if len(th) != 5:
            return CheckResult(
                self.name, ConstraintType.PREFERENCE, False, 0.0, False,
                _details(
                    "numeric.bad_threshold",
                    f"{self.aggregation}({self.attribute})={agg_val:.2f}  "
                    f"direction={self.direction}  threshold={self.threshold} "
                    f"(unsupported shape; expected 5-tuple [Q0, Q1, Q2, Q3, Q4] "
                    f"or 2-tuple [lo, hi] for auto-expansion)",
                    entity_type = self.entity_type,
                    attribute   = self.attribute,
                    aggregation = self.aggregation,
                    direction   = self.direction,
                    agg_value   = float(agg_val),
                    threshold   = list(self.threshold) if self.threshold else None,
                    n_values    = len(vals),
                ),
            )

        Q = [float(x) for x in th]
        qpos = self._quantile_position(agg_val, Q)
        if self.direction == "min":
            score  = 1.0 - qpos
            passed = agg_val <= Q[1]          # in the best 25%
            pass_threshold = Q[1]
        else:
            score  = qpos
            passed = agg_val >= Q[3]          # in the top 25%
            pass_threshold = Q[3]
        detail_msg = (f"{self.aggregation}({self.attribute})={agg_val:.2f}  "
                      f"direction={self.direction}  "
                      f"Q[0..4]={[round(q, 3) for q in Q]}  "
                      f"quantile_position={qpos:.3f}")
        return CheckResult(
            self.name, ConstraintType.PREFERENCE, passed, score, False,
            _details(
                "numeric",
                detail_msg,
                entity_type       = self.entity_type,
                attribute         = self.attribute,
                aggregation       = self.aggregation,
                direction         = self.direction,
                agg_value         = float(agg_val),
                quartiles         = Q,
                quantile_position = float(qpos),
                pass_threshold    = float(pass_threshold),
                n_values          = len(vals),
            ),
        )

    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporintg generation same as evaluation logic above
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 6. ScopedPreference  — restrict any preference to a subset of days/entities
# ─────────────────────────────────────────────────────────────────────────────

class ScopedPreference(Preference):
    """
    Wraps any inner Preference so it applies only to plan days / entities
    that satisfy a set of AtomicPreference filters.

    Parameters
    ----------
    inner         : Preference
    scope_filters : list[AtomicPreference]
                    Filters grouped internally by entity_type.
                    "Day" filters restrict which days are considered.
                    All other entity_type filters restrict which entities
                    on those days participate.
    require_match : bool   True → fail if no days match; False → trivial pass
    weight        : float | None
    name          : str

    Examples
    --------
    # Outdoor attractions only on stay-weekends
    ScopedPreference(
        inner=AtomicPreference("Attraction", "category", "in", ["Outdoor Activities"]),
        scope_filters=[
            AtomicPreference("Day", "week_group", "in", ["weekend"]),
            AtomicPreference("Day", "travel_phase", "==", "stay"),
        ],
    )

    # Concerts & Shows only when accommodation has "No visitors" rule
    ScopedPreference(
        inner=AtomicPreference("Attraction", "category", "in", ["Concerts & Shows"]),
        scope_filters=[
            AtomicPreference("Accommodation", "house_rules", "in", ["No visitors"]),
        ],
    )

    # Lex cuisine preference only for high-rated dinner restaurants
    ScopedPreference(
        inner=LexicographicPreference([french, italian], weight=2.0),
        scope_filters=[
            AtomicPreference("Restaurant", "meal_type", "==", "dinner"),
            AtomicPreference("Restaurant", "rating", ">=", 4.5),
        ],
    )
    """

    def __init__(self, inner: Preference,
                 scope_filters: list[AtomicPreference] | None = None,
                 require_match: bool = False,
                 weight: float | None = None,
                 name: str = ""):
        self.inner         = inner
        self.scope_filters = scope_filters or []
        self.require_match = require_match
        self.weight        = weight
        self.name          = name or f"Scoped:WHEN ({'; '.join(s.name for s in scope_filters)}) → {inner.name})"

    def _day_matches(self, day: dict) -> bool:
        for ap in self.scope_filters:
            et = ap.entity_type.lower()
            if et == "day":
                if not ap._check_one(day):
                    return False
            else:
                entities, _ = self._get_entities(day, et)
                if entities is None:
                    return False
                entity_list = entities if isinstance(entities, list) else [entities]
                if not any(ap._check_one(e) for e in entity_list):
                    return False
        return True

    def evaluate(self, plan: list) -> CheckResult:
        matching_days = {i for i, day in enumerate(plan) if self._day_matches(day)}
        n_total_days = len(plan)

        if not matching_days:
            if self.require_match:
                return CheckResult(
                    self.name, _ctype(self.weight), False, 0.0, False,
                    _details(
                        "scoped.no_match",
                        "No matching days",
                        n_scoped_days = 0,
                        n_total_days  = n_total_days,
                        require_match = self.require_match,
                    ),
                )
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "scoped.no_match_trivial",
                    "Trivially satisfied",
                    n_scoped_days = 0,
                    n_total_days  = n_total_days,
                    require_match = self.require_match,
                ),
            )

        # Entity-count bookkeeping for scope-filter selectivity.  Only
        # counts entities whose entity_type is a target of one of the
        # scope_filters (i.e. NOT the inner's target -- inner-side
        # entity counts are already carried on ``inner_result.details``).
        # "pre" is on matching days before the entity filter runs;
        # "post" is what survives the filter (== what inner will see
        # for those types).  Both are 0 when scope_filters carry only
        # ``Day`` filters (no entity-attribute narrowing at all).
        scope_entity_types: set[str] = set()
        for ap in self.scope_filters:
            et = ap.entity_type.lower()
            if et != "day":
                scope_entity_types.add(et)

        def _count_scope_entities(day: dict) -> int:
            n = 0
            for et in scope_entity_types:
                entities, _ = self._get_entities(day, et)
                if entities is None:
                    continue
                n += len(entities) if isinstance(entities, list) else 1
            return n

        n_entities_pre_scope_filter = sum(_count_scope_entities(plan[i])
                                           for i in matching_days)

        filtered_plan = []
        for i, day in enumerate(plan):
            if i not in matching_days:
                continue
            new_day = dict(day)
            for ap in self.scope_filters:
                et = ap.entity_type.lower()
                if et == "day":
                    continue
                entities, key = self._get_entities(day, et)
                if key is None:
                    continue
                if isinstance(entities, list):
                    new_day[key] = [e for e in entities if ap._check_one(e)]
                elif ap._check_one(entities):
                    new_day[key] = entities
                else:
                    del new_day[key]
            filtered_plan.append(new_day)

        n_entities_post_scope_filter = sum(_count_scope_entities(fd)
                                            for fd in filtered_plan)

        inner_result = self.inner.evaluate(filtered_plan)
        return CheckResult(
            self.name, _ctype(self.weight),
            inner_result.passed, inner_result.score, False,
            _details(
                "scoped",
                f"{len(filtered_plan)} scoped days",
                n_scoped_days = len(filtered_plan),
                n_total_days  = n_total_days,
                matching_day_indices = sorted(matching_days),
                n_entities_pre_scope_filter  = n_entities_pre_scope_filter,
                n_entities_post_scope_filter = n_entities_post_scope_filter,
                inner_passed  = bool(inner_result.passed),
                inner_score   = float(inner_result.score),
            ),
            [inner_result],
        )

    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporintg generation same as evaluation logic above
        raise NotImplementedError

    def _day_matches_inst(self, inst, instances: dict) -> bool:
        for ap in self.scope_filters:
            et = ap.entity_type
            if et.lower() == "day":
                if not ap._check_one(inst._fixed_attrs):
                    return False
            else:
                inst_list = instances.get(et) or instances.get(et.lower(), [])
                if not any(
                    i.day_idx == inst.day_idx and ap._check_one(i._fixed_attrs)
                    for i in inst_list
                ):
                    return False
        return True

    @staticmethod
    def _get_entities(day: dict, entity_type: str):
        """Return (entities, key) trying singular then plural key; (None, None) if absent."""
        for key in (entity_type, entity_type + "s"):
            if key in day:
                return day[key], key
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 7. CompensatoryPreference  (threshold-compensatory lexicographic semi-order)
# ─────────────────────────────────────────────────────────────────────────────

class CompensatoryPreference(Preference):
    """
    Threshold-compensatory lexicographic semi-order (Fishburn 1974).

    Primary and secondary criteria are expressed as AtomicPreferences,
    supporting numeric, string, and list-valued attributes uniformly.
    The secondary entity type may differ from the primary (cross-entity
    compensation). Currently compensated day-wise ONLY.

    Tier classification
    -------------------
    Tier-1  : primary_ap._check_one(entity) → True          full credit
    Tier-2  : margin_ap._check_one(entity)  → True          requires compensation
              AND NOT primary_ap satisfied
    Tier-3  : neither satisfied                              zero credit

    Compensation (Tier-2 only)
    --------------------------
    Same entity type  : secondary_ap._check_one(same entity)
    Cross entity type : any secondary-type entity on the same day satisfies
                        secondary_ap

    Parameters
    ----------
    primary_ap   : AtomicPreference   ideal condition on primary entity
                   e.g. AtomicPreference("Hotel", "rating", ">=", 4.0)
    margin_ap    : AtomicPreference   tolerance band — must share entity_type
                   e.g. AtomicPreference("Hotel", "rating", ">=", 3.5)
    secondary_ap : AtomicPreference   compensation condition — entity_type may differ
                   e.g. AtomicPreference("Hotel",      "price",    "<=",  80)
                        AtomicPreference("Hotel",      "amenities","in",  ["pool","spa"])
                        AtomicPreference("Restaurant", "cuisine",  "in",  ["local"])
    compensated_score  : float              partial credit for compensated Tier-2 (default 0.7)
    uncompensated_score  : float              partial credit for uncompensated Tier-2 (default 0.1)
    weight       : float | None
    name         : str

    Examples
    --------
    # Numeric primary, numeric secondary, same entity
    CompensatoryPreference(
        primary_ap   = AtomicPreference("Hotel", "rating", ">=", 4.0),
        margin_ap    = AtomicPreference("Hotel", "rating", ">=", 3.5),
        secondary_ap = AtomicPreference("Hotel", "price",  "<=", 80),
    )

    # Categorical compensation on same entity
    CompensatoryPreference(
        primary_ap   = AtomicPreference("Hotel", "stars",    ">=", 4),
        margin_ap    = AtomicPreference("Hotel", "stars",    ">=", 3),
        secondary_ap = AtomicPreference("Hotel", "amenities","in", ["pool", "spa"]),
    )

    # Cross-entity: hotel quality compensated by restaurant cuisine on same day
    CompensatoryPreference(
        primary_ap   = AtomicPreference("Hotel",      "rating",  ">=", 4.0),
        margin_ap    = AtomicPreference("Hotel",      "rating",  ">=", 3.5),
        secondary_ap = AtomicPreference("Restaurant", "cuisine", "in", ["local", "fine-dining"]),
    )
    """

    def __init__(
        self,
        primary_ap:   AtomicPreference,
        margin_ap:    AtomicPreference,
        secondary_ap: AtomicPreference,
        compensated_score:  float        = 0.7,
        uncompensated_score:  float      = 0.1,
        weight:       float | None = 1.0,
        name:         str          = "",
    ):
        assert primary_ap.entity_type == margin_ap.entity_type, (
            f"primary_ap and margin_ap must share entity_type; "
            f"got {primary_ap.entity_type!r} vs {margin_ap.entity_type!r}"
        )
        assert 0 < compensated_score < 1, "compensated_score must be in (0, 1)"

        self.primary_ap    = primary_ap
        self.margin_ap     = margin_ap
        self.secondary_ap  = secondary_ap
        self.compensated_score   = compensated_score
        self.uncompensated_score   = uncompensated_score
        self.weight        = weight
        self._cross_entity = (primary_ap.entity_type != secondary_ap.entity_type)

        self.name = name or (
            f"Compensatory:({primary_ap.name} "
            f"[margin:{margin_ap.name}] "
            f"→ compensate:{secondary_ap.name})"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _from_day(day: dict, entity_type: str) -> list:
        """Collect entities of a given type from a single day dict."""
        et = entity_type.lower()
        for key in (et, et + "s", entity_type, entity_type + "s"):
            val = day.get(key)
            if val is not None:
                return val if isinstance(val, list) else [val]
        return []

    # ── tier classification ────────────────────────────────────────────────────

    def _classify_day(self, day: dict) -> list[tuple[str, dict]]:
        """
        Classify each primary entity in a single day.

        Returns list of (tier_label, entity_dict) where
        tier_label ∈ {"tier1", "tier2_comp", "tier2_uncomp", "tier3"}.

        Cross-entity compensation is resolved at day scope: a Tier-2 primary
        entity is compensated if ANY secondary entity on the same day satisfies
        secondary_ap.
        """
        primary_entities = self._from_day(day, self.primary_ap.entity_type)
        if not primary_entities:
            return []

        # Resolve cross-entity compensation once per day (not per entity)
        if self._cross_entity:
            secondary_entities = self._from_day(day, self.secondary_ap.entity_type)
            day_compensated    = any(
                self.secondary_ap._check_one(se) for se in secondary_entities
            )

        results = []
        for pe in primary_entities:
            if self.primary_ap._check_one(pe):
                results.append(("tier1", pe))
            elif self.margin_ap._check_one(pe):
                if self._cross_entity:
                    compensated = day_compensated
                else:
                    compensated = self.secondary_ap._check_one(pe)
                results.append(("tier2_comp" if compensated else "tier2_uncomp", pe))
            else:
                results.append(("tier3", pe))

        return results

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, plan: list) -> CheckResult:
        all_classified = []
        for day in plan:
            all_classified.extend(self._classify_day(day))

        if not all_classified:
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "compensatory.no_primary",
                    f"No {self.primary_ap.entity_type} entities found — "
                    f"trivially satisfied",
                    primary_entity_type = self.primary_ap.entity_type,
                    cross_entity        = self._cross_entity,
                ),
            )

        counts = {"tier1": 0, "tier2_comp": 0, "tier2_uncomp": 0, "tier3": 0}
        for label, _ in all_classified:
            counts[label] += 1

        n_total = len(all_classified)
        n_t1    = counts["tier1"]
        n_t2c   = counts["tier2_comp"]
        n_t2u   = counts["tier2_uncomp"]
        n_t3    = counts["tier3"]

        # Tier-1 > Tier-2 compensated > Tier-2 uncompensated > Tier-3
        score  = (
            n_t1 * 1.0 + n_t2c * self.compensated_score + n_t2u * self.uncompensated_score
        ) / n_total
        passed = (n_t3 == 0) and (n_t2u == 0)

        details_msg = (
            f"T1={n_t1}  T2_ok={n_t2c}  T2_fail={n_t2u}  T3={n_t3}  "
            f"score={score:.2f}"
        )

        sub = []
        for label, e in all_classified:
            name_e = e.get("name", self.primary_ap.entity_type)
            p_val  = e.get(self.primary_ap.attribute, "?")
            s_val  = (
                "cross-entity" if self._cross_entity
                else e.get(self.secondary_ap.attribute, "?")
            )
            common_fields = {
                "tier":               label,
                "entity_name":        name_e,
                "primary_attribute":  self.primary_ap.attribute,
                "primary_value":      p_val,
                "secondary_attribute": self.secondary_ap.attribute,
                "secondary_value":    s_val,
                "cross_entity":       self._cross_entity,
            }
            if label == "tier1":
                sub.append(CheckResult(
                    name_e, ConstraintType.PREFERENCE, True, 1.0, False,
                    _details(
                        "compensatory.entity",
                        f"Tier-1: {self.primary_ap.attribute}={p_val} satisfies ideal",
                        **common_fields,
                    ),
                ))
            elif label == "tier2_comp":
                sub.append(CheckResult(
                    name_e, ConstraintType.PREFERENCE, True, self.compensated_score, False,
                    _details(
                        "compensatory.entity",
                        f"Tier-2✓: {self.primary_ap.attribute}={p_val} in band, "
                        f"{self.secondary_ap.attribute}={s_val} compensates",
                        **common_fields,
                    ),
                ))
            elif label == "tier2_uncomp":
                sub.append(CheckResult(
                    name_e, ConstraintType.PREFERENCE, False, self.uncompensated_score, False,
                    _details(
                        "compensatory.entity",
                        f"Tier-2✗: {self.primary_ap.attribute}={p_val} in band but "
                        f"{self.secondary_ap.attribute}={s_val} insufficient",
                        **common_fields,
                    ),
                ))
            else:
                sub.append(CheckResult(
                    name_e, ConstraintType.PREFERENCE, False, 0.0, False,
                    _details(
                        "compensatory.entity",
                        f"Tier-3✗: {self.primary_ap.attribute}={p_val} outside margin",
                        **common_fields,
                    ),
                ))

        # Tier-1 (ideal) and Tier-2-compensated both count as "passing"
        # entities under the compensatory semi-order.
        n_passed_entities = n_t1 + n_t2c
        return CheckResult(
            self.name, _ctype(self.weight), passed, score, False,
            _details(
                "compensatory",
                details_msg,
                primary_ap_name     = self.primary_ap.name,
                margin_ap_name      = self.margin_ap.name,
                secondary_ap_name   = self.secondary_ap.name,
                cross_entity        = self._cross_entity,
                n_entities          = n_total,
                passed_entities     = (n_passed_entities / n_total) if n_total else 0,
                tier_pass_counts    = counts,
                compensated_score   = self.compensated_score,
                uncompensated_score = self.uncompensated_score,
            ),
            sub,
        )

    # ── Z3 generation ─────────────────────────────────────────────────────────

    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporintg generation same as evaluation logic above
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TemporalPreference — all 9 modalities from PDDL
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPreference(Preference):
    """
    Temporal preferences covering nine PDDL3 temporal constraints / operators.

    Temporal Constraints / Operators (PDDL3 semantics)
    --------------------------------------------------
    always           : every day has at least one subject entity.
                       reference_ap, time_start, time_end unused.
    sometime         : at least one subject entity exists anywhere.
                       reference_ap, time_start, time_end unused.
    atmost_once      : subject appears at most once.
                       reference_ap, time_start, time_end unused.
    within(time_end) : subject appears within the first `time_end` positions
                       (or days, depending on day_as_step) of group.
                       reference_ap and time_start unused.
    sometime_before  : subject occurs before reference
                       (strict=True → ALL subject precede ALL reference per group).
                       time_start, time_end unused.
    sometime_after   : subject occurs after reference
                       (strict=True → ALL subject follow ALL reference per group).
                       time_start, time_end unused.
    always_within(time_end) :
                       whenever subject occurs, reference must also occur
                       within the next `time_end` positions after subject.
                       time_start unused.
    hold_during(time_start, time_end) :
                       subject property holds on every day d ∈ [time_start, time_end].
                       For every day in the window, at least one entity of
                       subject's entity_type must exist on that day AND all
                       such entities must satisfy subject_ap. A day with no
                       entity of the relevant type FAILS that day (no vacuous
                       satisfaction — matches PDDL 3.0). reference_ap unused.
    hold_after(time_start) :
                       subject property holds on every day d ≥ time_start
                       through the last plan day. Same strict per-day
                       semantics as hold_during. reference_ap, time_end unused.

    Note the deliberate asymmetry: for within / always_within, time_end is a
    *count* (relative window size); for hold_during / hold_after, time_start
    and time_end are *absolute day indices*.

    Parameters
    ----------
    subject_ap   : AtomicPreference   φ1 — the entity / property the claim is about
    reference_ap : AtomicPreference | None
                   φ2 — the related entity. Required by: sometime_before,
                   sometime_after, always_within. Unused by everything else.
    op           : str   one of the nine temporal constraints / operators
    time_start   : int | None   see table above
    time_end     : int | None   see table above
    day_as_step  : bool
        True (default) — ordering uses day index only (within / always_within
                         only; hold_during / hold_after are inherently
                         day-indexed by construction).
        False          — ordering uses slot-level sequence positions.
    scope        : str   grouping key — "global" | "per_day" | "week_group" | "per_city" ...
    strict       : bool  for sometime_before / sometime_after only
    weight       : float | None
    name         : str
    """

    _SINGLE_PRED_CONSTRAINTS       = frozenset(["always", "sometime",
                                                "atmost_once", "within"])
    _TWO_PRED_CONSTRAINTS          = frozenset(["sometime_before",
                                                "sometime_after",
                                                "always_within"])
    _TIMED_SINGLE_PRED_CONSTRAINTS = frozenset(["hold_during", "hold_after"])
    TEMPORAL_CONSTRAINTS = (_SINGLE_PRED_CONSTRAINTS
                            | _TWO_PRED_CONSTRAINTS
                            | _TIMED_SINGLE_PRED_CONSTRAINTS)

    # Declarative requirements for time bounds.
    _TIME_END_REQUIRED   = frozenset(["within", "always_within", "hold_during"])
    _TIME_START_REQUIRED = frozenset(["hold_during", "hold_after"])

    def __init__(
        self,
        subject_ap:   AtomicPreference,
        reference_ap: AtomicPreference | None = None,
        op:           str           = "always",
        time_start:   int | None    = None,
        time_end:     int | None    = None,
        day_as_step:  bool          = True,
        scope:        str           = "global",
        strict:       bool          = False,
        weight:       float | None  = None,
        name:         str           = "",
    ):
        assert op in self.TEMPORAL_CONSTRAINTS, f"Unknown temporal operator {op!r}"

        if op in self._TWO_PRED_CONSTRAINTS:
            assert reference_ap is not None, (
                f"{op!r} requires reference_ap; single-predicate operators "
                f"(no reference_ap) are: "
                f"{sorted(self._SINGLE_PRED_CONSTRAINTS | self._TIMED_SINGLE_PRED_CONSTRAINTS)}"
            )

        if op in self._TIME_START_REQUIRED:
            role = "window start day" if op == "hold_during" else "anchor day"
            assert time_start is not None, f"{op!r} requires time_start ({role})"
        if op in self._TIME_END_REQUIRED:
            role = ("window size" if op in ("within", "always_within")
                    else "window end day")
            assert time_end is not None, f"{op!r} requires time_end ({role})"
        if op == "hold_during":
            assert time_start <= time_end, (
                f"hold_during: time_start ({time_start}) must be "
                f"<= time_end ({time_end})"
            )

        self.subject_ap   = subject_ap
        self.reference_ap = reference_ap
        self.op           = op
        self.time_start   = time_start
        self.time_end     = time_end
        self.scope        = scope
        self.day_as_step  = day_as_step
        if scope == "per_day":
            # day_as_step=True with scope='per_day' is meaningless: all
            # entities within a per_day group share the same day index.
            self.day_as_step = False
        self.strict = strict
        self.weight = weight

        # Name suffix per op family
        if op in ("within", "always_within"):
            suffix = f"(time_end={time_end})"
        elif op == "hold_during":
            suffix = f"(time_start={time_start},time_end={time_end})"
        elif op == "hold_after":
            suffix = f"(time_start={time_start})"
        else:
            suffix = ""
        ref_str = (f", {reference_ap.name}"
                   if reference_ap and op in self._TWO_PRED_CONSTRAINTS
                   else "")
        self.name = name or (
            f"Temporal:{op}{suffix}[{scope}, step={'day' if self.day_as_step else 'slot'}]({subject_ap.name}; {ref_str})"
        )

    # ── sequence helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _seq_matches(entity: dict, ap: AtomicPreference) -> bool:
        if entity.get("entity_type", "").lower() != ap.entity_type.lower():
            return False
        return ap._check_one(entity)

    def _group_key(self, entity: dict) -> str:
        if self.scope == "global":  return "_all_"
        if self.scope == "per_day": return str(entity.get("_day", "_"))
        if self.scope == "per_city": return str(entity.get("_city", "_"))
        return str(entity.get(self.scope, "_none_"))

    def _group_sequences(self, seq: list) -> dict[str, list[tuple[int, dict]]]:
        groups: dict[str, list] = defaultdict(list)
        for i, e in enumerate(seq):
            groups[self._group_key(e)].append((i, e))
        return dict(groups)

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, plan: list) -> CheckResult:
        seq = _extract_sequence(plan)

        if not seq:
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "temporal.empty_plan",
                    "Empty plan. Rely on other external metrics as a measure "
                    "as these measures are anyway not possible — trivially "
                    "satisfied.",
                    op    = self.op,
                    scope = self.scope,
                ),
            )

        groups = self._group_sequences(seq)

        if self.op in self._SINGLE_PRED_CONSTRAINTS:
            return self._aggregate([
                self._eval_single_pred_group(indexed, gkey)
                for gkey, indexed in sorted(groups.items())
            ])

        if self.op in self._TIMED_SINGLE_PRED_CONSTRAINTS:
            return self._aggregate([
                self._eval_timed_group(indexed, gkey)
                for gkey, indexed in sorted(groups.items())
            ])

        return self._aggregate([
            self._eval_group(indexed, gkey)
            for gkey, indexed in sorted(groups.items())
        ])

    # ── always / sometime / atmost_once / within ─────────────────────────────

    def _eval_single_pred_group(
        self,
        indexed: list[tuple[int, dict]],
        gkey:    str,
    ) -> CheckResult:
        """
        Evaluate always / sometime / atmost_once / within in one scope group.

        always              : every day in this group has at least one subject
        sometime            : at least one day in this group has at least one subject
        atmost_once         : at most one subject instance exists across this group
        within(time_end)    : subject must appear within the first `time_end`
                              positions of the group (delegated to _eval_within)
        """
        label    = f"[{self.scope}={gkey}]"
        entities = [e for _, e in indexed]

        if self.op in ("always", "sometime"):
            day_ents: dict[int, list] = defaultdict(list)
            for e in entities:
                day_ents[e.get("_day", 0)].append(e)

            days_with_subject = sum(
                any(self._seq_matches(e, self.subject_ap) for e in ents)
                for ents in day_ents.values()
            )
            total = len(day_ents)

            if self.op == "always":
                passed = days_with_subject == total
                score  = days_with_subject / total
                return CheckResult(
                    self.name, _ctype(self.weight), passed, score, False,
                    _details(
                        "temporal.group",
                        f"{label} {days_with_subject}/{total} days have subject",
                        op                = "always",
                        scope             = self.scope,
                        group_key         = gkey,
                        days_total        = total,
                        passed_days       = (days_with_subject / total) if total else 0,
                    ),
                )

            else:  # sometime
                passed = days_with_subject > 0
                score  = float(passed)
                return CheckResult(
                    self.name, _ctype(self.weight), passed, score, False,
                    _details(
                        "temporal.group",
                        f"{label} {days_with_subject}/{total} days have subject",
                        op                = "sometime",
                        scope             = self.scope,
                        group_key         = gkey,
                        days_total        = total,
                        passed_days       = (days_with_subject / total) if total else 0,
                    ),
                )

        if self.op == "atmost_once":
            subject_idxs = [j for j, e in enumerate(entities)
                            if self._seq_matches(e, self.subject_ap)]
            n      = len(subject_idxs)
            L      = max(len(entities), 1)
            passed = n <= 1
            score  = 1.0 if passed else max(0.0, 1.0 - (n - 1) / L)
            satisfaction_suffix = "" if n else " — trivially satisfied"
            # To ensure non-trivial solution, pair with `sometime` to enforce
            # exactly-once semantics.
            return CheckResult(
                self.name, _ctype(self.weight), passed, score, not bool(n),
                _details(
                    "temporal.group",
                    f"{label} subject appears {n} time(s) in group of length {L}"
                    f"{satisfaction_suffix}",
                    op            = "atmost_once",
                    scope         = self.scope,
                    group_key     = gkey,
                    subject_count = n,
                    group_length  = L,
                ),
            )

        if self.op == "within":
            return self._eval_within(indexed, gkey)

        return CheckResult(
            self.name, _ctype(self.weight), True, 1.0, True,
            _details(
                "temporal.unknown_op",
                f"{label} unknown op {self.op!r} — trivially satisfied",
                op        = self.op,
                scope     = self.scope,
                group_key = gkey,
            ),
        )

    def _eval_within(
        self, indexed: list[tuple[int, dict]], gkey: str
    ) -> CheckResult:
        """
        within(time_end): subject must appear within the first `time_end`
        LOCAL positions of this group. Position clock resets per group.
        reference_ap and time_start unused.
        """
        label         = f"[{self.scope}={gkey}]"
        subject_local = [j for j, (_, e) in enumerate(indexed)
                         if self._seq_matches(e, self.subject_ap)]

        if not subject_local:
            return CheckResult(
                self.name, _ctype(self.weight), False, 0.0, False,
                _details(
                    "temporal.within",
                    f"{label} No subject — hence within fails",
                    op          = "within",
                    scope       = self.scope,
                    group_key   = gkey,
                    no_subject  = True,
                    time_end    = self.time_end,
                    day_as_step = self.day_as_step,
                ),
            )

        if self.day_as_step:
            distinct_days = sorted({e.get("_day", 0) for _, e in indexed})
            L = len(distinct_days)
            step_pos = [distinct_days.index(e.get("_day", 0)) for _, e in indexed]
        else:
            step_pos = list(range(len(indexed)))
            L        = max(len(indexed), 1)

        first_subject_pos = step_pos[subject_local[0]]  # zero-indexed
        passed            = first_subject_pos < self.time_end

        if passed:
            score = 1.0
        else:
            # "Just missed" → score near 1.0; "appeared at very end" → 0.0.
            # +1 in denominator guards against time_end == L (no slots left).
            excess    = first_subject_pos - (self.time_end - 1)
            remaining = L - self.time_end
            score     = max(0.0, 1.0 - excess / (remaining + 1))

        return CheckResult(
            self.name, _ctype(self.weight), passed, score, False,
            _details(
                "temporal.within",
                f"{label} first subject at "
                f"{'day' if self.day_as_step else 'slot'} pos {first_subject_pos}, "
                f"time_end={self.time_end}, L={L}",
                op                = "within",
                scope             = self.scope,
                group_key         = gkey,
                first_subject_pos = int(first_subject_pos),
                time_end          = self.time_end,
                group_length      = int(L),
                day_as_step       = self.day_as_step,
                no_subject        = False,
            ),
        )

    # ── hold_during / hold_after ─────────────────────────────────────────────

    def _eval_timed_group(
        self,
        indexed: list[tuple[int, dict]],
        gkey:    str,
    ) -> CheckResult:
        """
        Evaluate hold_during / hold_after in one scope group.

        Semantics (PDDL 3.0-faithful, no vacuous satisfaction):
            hold_during(time_start, time_end):
                ∀ d ∈ [time_start, time_end]:
                    ∃ entity e on day d with e.type == subject.entity_type
                    AND ∀ such e: subject_ap(e) holds.

            hold_after(time_start):
                ∀ d ∈ [time_start, max_plan_day]: (same condition)

        A day with NO entity of subject's entity_type FAILS that day — this
        matches PDDL 3.0, where φ must hold in the world state at every
        time t in the window with no exemption for "no relevant entity
        scheduled."

        Scoring:
            day_score(d) = matching / total entities of subject.entity_type on d
                         = 0.0 if no entity of subject's type on day d (fails)
            score        = mean(day_score) across all window days
            passed       = all day_scores == 1.0
            trivial      = False (timed window constraints are never vacuous
                           under PDDL 3.0 semantics; empty plans are handled
                           in evaluate())

        Out-of-plan window days are silently dropped — plan-length / delivery
        compliance is measured by external metrics like delivery rate.
        """
        label    = f"[{self.scope}={gkey}]"
        entities = [e for _, e in indexed]
        etype    = self.subject_ap.entity_type.lower()

        day_ents:  dict[int, list] = defaultdict(list)
        plan_days: set[int]        = set()
        for e in entities:
            d = e.get("_day", 0)
            plan_days.add(d)
            if e.get("entity_type", "").lower() == etype:
                day_ents[d].append(e)

        if not plan_days:
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "temporal.timed",
                    f"{label} no days in group — trivially satisfied",
                    op        = self.op,
                    scope     = self.scope,
                    group_key = gkey,
                    plan_days = [],
                ),
            )

        max_plan_day = max(plan_days)

        # `time_start` and `time_end` are specified 1-indexed in the
        # preference bank (day 1 = first trip day), but the sequence's
        # internal `_day` is 0-indexed (from ``enumerate(plan)``).  Shift
        # each bound down by one when constructing the window; the
        # display repr keeps the human-readable 1-indexed form.
        # (within / always_within use `time_end` as a *count* -- window
        # size -- so no shift is needed there; only these two ops treat
        # `time_start` / `time_end` as absolute day indices.)
        if self.op == "hold_during":
            ts0, te0 = self.time_start - 1, self.time_end - 1
            window = sorted(d for d in range(ts0, te0 + 1)
                            if d in plan_days)
            window_repr = f"[{self.time_start},{self.time_end}]"
        else:  # hold_after
            ts0 = self.time_start - 1
            window = sorted(d for d in range(ts0, max_plan_day + 1)
                            if d in plan_days)
            window_repr = f"[{self.time_start},..]"

        if not window:
            # Window does not overlap any plan day — defer to delivery metrics.
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "temporal.timed",
                    f"{label} {self.op} window {window_repr} has no overlap "
                    f"with plan days {sorted(plan_days)} — deferred to "
                    f"delivery metrics",
                    op           = self.op,
                    scope        = self.scope,
                    group_key    = gkey,
                    window_repr  = window_repr,
                    plan_days    = sorted(plan_days),
                    time_start   = self.time_start,
                    time_end     = self.time_end,
                    overlap      = False,
                ),
            )

        day_scores:   list[float] = []
        failing_days: list[int]   = []
        for d in window:
            ents = day_ents.get(d, [])
            if not ents:
                day_scores.append(0.0)
                failing_days.append(d)
                continue
            matching = sum(1 for e in ents if self.subject_ap._check_one(e))
            ds       = matching / len(ents)
            day_scores.append(ds)
            if ds < 1.0:
                failing_days.append(d)

        score   = sum(day_scores) / len(day_scores)
        passed  = all(s == 1.0 for s in day_scores)
        trivial = False

        scores_str = ", ".join(f"d{d}:{s:.2f}"
                               for d, s in zip(window, day_scores))
        detail_msg = (f"{label} {self.op}{window_repr} window={window} "
                      f"day_scores={{{scores_str}}}")
        if failing_days:
            detail_msg += f" failing_days={failing_days}"
        n_days = len(window)
        n_perfect_days = n_days - len(failing_days)
        return CheckResult(
            self.name, _ctype(self.weight), passed, score, trivial,
            _details(
                "temporal.timed",
                detail_msg,
                op           = self.op,
                scope        = self.scope,
                group_key    = gkey,
                time_start   = self.time_start,
                time_end     = self.time_end,
                window_repr  = window_repr,
                window       = list(window),
                day_scores   = {int(d): float(s)
                                for d, s in zip(window, day_scores)},
                failing_days = list(failing_days),
                n_days       = n_days,
                passed_days  = (n_perfect_days / n_days) if n_days else 0,
                overlap      = True,
            ),
        )

    # ── sometime_before / sometime_after / always_within ─────────────────────

    def _eval_group(
        self,
        indexed: list[tuple[int, dict]],
        gkey:    str,
    ) -> CheckResult:
        """
        Evaluate one scope group for the two-predicate operators:
        sometime_before, sometime_after, always_within.
        hold_during / hold_after are handled in _eval_timed_group.
        """
        label           = f"[{self.scope}={gkey}]"
        global_idxs     = [i for i, _ in indexed]
        entities        = [e for _, e in indexed]

        subject_local   = [j for j, e in enumerate(entities)
                           if self._seq_matches(e, self.subject_ap)]
        reference_local = [j for j, e in enumerate(entities)
                           if self._seq_matches(e, self.reference_ap)]

        subject_global   = [global_idxs[j] for j in subject_local]
        reference_global = [global_idxs[j] for j in reference_local]

        # day_as_step collapses positions to day indices: within-day ordering
        # becomes a tie; cross-day ordering is preserved.
        if self.day_as_step:
            step_pos = [ent.get("_day", 0) for ent in entities]
        else:
            step_pos = list(range(len(entities)))

        if not subject_local:
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "temporal.pair",
                    f"{label} No subject — trivially satisfied",
                    op         = self.op,
                    scope      = self.scope,
                    group_key  = gkey,
                    no_subject = True,
                    no_reference = not bool(reference_local),
                ),
            )
        if not reference_local:
            return CheckResult(
                self.name, _ctype(self.weight), False, 0.0, False,
                _details(
                    "temporal.pair",
                    f"{label} No reference but subject exists — {self.op} fails",
                    op           = self.op,
                    scope        = self.scope,
                    group_key    = gkey,
                    no_subject   = False,
                    no_reference = True,
                ),
            )

        # sometime_before / sometime_after
        if self.op in ("sometime_before", "sometime_after"):
            if self.op == "sometime_before":
                pairs_ok = sum(1 for si in subject_local
                               for ri in reference_local
                               if step_pos[si] < step_pos[ri])
            else:
                pairs_ok = sum(1 for ri in reference_local
                               for si in subject_local
                               if step_pos[ri] < step_pos[si])
            total  = max(len(subject_local) * len(reference_local), 1)
            passed = (pairs_ok == total) if self.strict else (pairs_ok > 0)
            score  = pairs_ok / total if self.strict else float(passed)
            return CheckResult(
                self.name, _ctype(self.weight), passed, score, False,
                _details(
                    "temporal.pair",
                    f"{label} subject={subject_global} "
                    f"reference={reference_global} "
                    f"strict={self.strict} "
                    f"day_as_step={self.day_as_step}",
                    op                  = self.op,
                    scope               = self.scope,
                    group_key           = gkey,
                    subject_positions   = list(subject_global),
                    reference_positions = list(reference_global),
                    pairs_total         = int(total),
                    passed_pairs        = (int(pairs_ok) / int(total)) if total else 0,
                    strict              = self.strict,
                    day_as_step         = self.day_as_step,
                ),
            )

        # always_within: every subject must have reference within next time_end
        if self.op == "always_within":
            subj_satisfied = [
                si for si in subject_local
                if any(0 <= (step_pos[ri] - step_pos[si]) <= self.time_end
                       for ri in reference_local)
            ]
            score  = len(subj_satisfied) / len(subject_local)
            passed = len(subj_satisfied) == len(subject_local)
            n_subjects = len(subject_local)
            return CheckResult(
                self.name, _ctype(self.weight), passed, score, False,
                _details(
                    "temporal.pair",
                    f"{label} {len(subj_satisfied)}/{len(subject_local)} "
                    f"subject triggers have reference within next "
                    f"time_end={self.time_end} steps",
                    op              = "always_within",
                    scope           = self.scope,
                    group_key       = gkey,
                    n_subjects      = n_subjects,
                    passed_subjects = (len(subj_satisfied) / n_subjects)
                                       if n_subjects else 0,
                    time_end        = self.time_end,
                    day_as_step     = self.day_as_step,
                ),
            )

        return CheckResult(
            self.name, _ctype(self.weight), True, 1.0, True,
            _details(
                "temporal.unknown_op",
                f"{label} unknown op {self.op!r} — trivially satisfied",
                op        = self.op,
                scope     = self.scope,
                group_key = gkey,
            ),
        )

    # ── aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, group_results: list) -> CheckResult:
        if not group_results:
            return CheckResult(
                self.name, _ctype(self.weight), True, 1.0, True,
                _details(
                    "temporal.aggregate",
                    "No qualifying group — trivially satisfied",
                    op            = self.op,
                    scope         = self.scope,
                    n_groups      = 0,
                    passed_groups = 0,
                ),
            )
        passed  = all(r.passed for r in group_results)
        score   = sum(r.score for r in group_results) / len(group_results)
        n_pass  = sum(1 for r in group_results if r.passed)
        # NOTE the deliberate asymmetry with ``passed`` above: ``passed``
        # is ALL-quantified, ``trivial`` is ANY-quantified.  Under a
        # grouped scope (``per_day`` / ``per_city`` / ...) the preference
        # is decomposed into one independent obligation per group, and
        # some groups routinely carry no subject entity at all.  The flag
        # therefore reads as "CONTAINS vacuous groups", not "was vacuous
        # throughout" -- it marks that the preference's effective sample
        # was smaller than ``n_groups``, which is the useful signal for a
        # grouped operator.
        #
        # CONSEQUENCE, and the reason this is spelled out: ``trivial`` no
        # longer implies ``passed``.  A preference whose day 1 genuinely
        # failed and whose days 0 and 2 had no subject comes back
        # ``passed=False, trivial=True``.  Downstream consumers that use
        # the flag as a BINARY PARTITION (see
        # ``evaluation/analyze_performance.py::_bucket_prefs``, where
        # trivial_n + nontrivial_n == n) will therefore book such a row
        # as fully trivial and exclude it from non-trivial statistics,
        # slightly deflating the trivial pass rate and hiding a genuine
        # failure from the non-trivial one.  Measured across all five
        # models and both published splits this affects 15 preferences,
        # all Temporal (10 always_within, 5 sometime_before), moving
        # non-trivial pass rates by 0.1-0.3pp.  Kept as-is by design;
        # ``all(...)`` here would restore trivial => passed but would
        # lose the partial-vacuity signal.
        trivial = any(r.trivial for r in group_results)
        return CheckResult(
            self.name, _ctype(self.weight), passed, score, trivial,
            _details(
                "temporal.aggregate",
                f"{n_pass}/{len(group_results)} [{self.scope}] groups pass",
                op            = self.op,
                scope         = self.scope,
                n_groups      = len(group_results),
                passed_groups = (int(n_pass) / len(group_results))
                                 if group_results else 0,
            ),
            group_results,
        )

    # ── Z3 generation ────────────────────────────────────────────────────────

    def to_z3(self, instances: dict, solver) -> None:
        # TODO: implement supporting generation same as evaluation logic above
        raise NotImplementedError


def _all_entities(day: dict) -> list[dict]:
    """Collect all entity dicts from a day dict."""
    result = []
    for key in ("restaurants", "attractions", "transportation"):
        result.extend(day.get(key, []))
    acc = day.get("accommodation")
    if acc:
        result.append(acc)
    return result


def _filter_str(f: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in f.items())


def _z3_available() -> bool:
    try:
        import z3  # noqa
        return True
    except ImportError:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ctype(weight) -> ConstraintType:
    return ConstraintType.HARD if weight is None else ConstraintType.PREFERENCE


def _entity_type_of(pref) -> str | None:
    """Return the primary entity_type a preference targets, or None."""
    if pref is None:
        return None
    if isinstance(pref, AtomicPreference):
        return pref.entity_type
    # LexicographicPreference
    if hasattr(pref, "preferences"):
        for child in pref.preferences:
            et = _entity_type_of(child)
            if et:
                return et
    # CompositePreference
    if hasattr(pref, "children"):
        for child in pref.children:
            et = _entity_type_of(child)
            if et:
                return et
    # ConditionalPreference
    if hasattr(pref, "then_pref") and pref.then_pref:
        return _entity_type_of(pref.then_pref)
    return None


def _collect(plan: list, entity_type: str) -> list[dict]:
    """
    Flatten all occurrences of entity_type from a plan (list of day dicts).

    "Day" is a virtual entity type: each day dict is returned as-is,
    enriched with entity_type="day", so AtomicPreference conditions on
    Day attributes (week_group, travel_phase) can be checked uniformly.
    """
    et = entity_type.lower()
    results = []

    if et == "day":
        for day in plan:
            results.append({**day, "entity_type": "day"})
        return results

    for day in plan:
        for key in (entity_type, et, entity_type + "s", et + "s"):
            val = day.get(key)
            if isinstance(val, list):
                results.extend(val)
            elif isinstance(val, dict):
                results.append(val)

    return results


def _extract_sequence(plan: list) -> list[dict]:
    """
    Return a flat ordered list of all activities in the plan.

    Each item carries its own attributes plus day-level metadata:
      _day        : int   day index (0-based)
      _slot       : int   slot within day
      _day_number : int   human day number (1-based, from plan dict)
      week_group  : str   "weekday" | "weekend" (from day dict if present)
      travel_phase : str   "arrival" | "stay" | "departure"
      city        : str   city name if present in day dict
      entity_type : str   lowercased entity type

    These day-level fields enable arbitrary scope grouping in
    TemporalOrderPreference without requiring a separate lookup.
    """
    seq = []
    for d, day in enumerate(plan):
        # Day-level context carried into every entity on this day
        day_ctx = {
            "_day":        d,
            "_day_number": day.get("day", d + 1),
            "week_group":  day.get("week_group", "weekday"),
            "travel_phase": day.get("travel_phase", "stay"),
            "city":        day.get("city", ""),
        }
        # `transportation` covers every intercity mode -- flight,
        # self-driving, taxi -- under one uniform key.  ``.rstrip("s")``
        # leaves it unchanged (no trailing s), so items are surfaced
        # with entity_type="transportation" -- which is what the
        # preference bank's Temporal subject entity uses.
        for key in ("attractions", "restaurants", "transportation"):
            for s, item in enumerate(day.get(key, [])):
                seq.append({**day_ctx, **item, "_slot": s,
                             "entity_type": key.rstrip("s")})
        acc = day.get("accommodation")
        if isinstance(acc, dict):
            seq.append({**day_ctx, **acc, "_slot": 999,
                        "entity_type": "accommodation"})
    return seq


def _matches(entity: dict, filter_: dict) -> bool:
    for k, v in filter_.items():
        if k == "entity_type":
            if entity.get("entity_type", "") != v:
                return False
        else:
            actual = entity.get(k)
            if isinstance(v, list):
                if str(actual) not in [str(x) for x in v]:
                    return False
            else:
                if str(actual) != str(v):
                    return False
    return True


def _filter_str(f: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in f.items())


# ─────────────────────────────────────────────────────────────────────────────
# AttributeSpec & EntitySpec (for Z3 generation support)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributeSpec:
    name:   str
    domain: Domain

    def make_var(self, instance_name: str):
        return self.domain.make_z3_var(f"{instance_name}__{self.name}")


@dataclass
class EntitySpec:
    name:       str
    attributes: dict[str, AttributeSpec]
