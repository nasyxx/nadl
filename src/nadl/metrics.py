#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Nov 29, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : metrics.py
project  : nadl
license  : GPL-3.0+

* Metric class and functions
* Some simple metrics that not in optax loss.  For complex metrics, use sklearn.
"""
# ruff: noqa: F722

from functools import partial
import operator
from abc import abstractmethod
from collections.abc import Callable
import sklearn.metrics as m
import jax
import jax.numpy as jnp
from equinox import (
  AbstractVar,
  Module,
  combine,
  field,
  is_array,
  partition,
  tree_at,
  tree_equal,
  tree_pformat,
  filter_vmap,
  filter_pure_callback,
)

from jaxtyping import Array, Int, Num
from typing import Literal, Self

from .utils import filter_concat

type N = Num[Array, "#a"]


def convert(x: float | Num | None) -> N:
  """Convert to float."""
  if x is None:
    return jnp.asarray(jnp.nan).reshape(-1)
  if isinstance(x, jax.Array):
    return x.reshape(-1)
  return jnp.asarray(x).reshape(-1)


def roc_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> N:
  """Compute ROC."""
  return jax.pure_callback(
    m.roc_auc_score, jax.ShapeDtypeStruct((), jnp.float32), labels, preds
  )


def average_precision_score(
  labels: Num[Array, " A"],
  preds: Num[Array, " A"],
  average: Literal["micro", "macro"] = "macro",
) -> N:
  """Compute PR."""
  return jax.pure_callback(
    partial(m.average_precision_score, average=average),
    jax.ShapeDtypeStruct((), jnp.float32),
    labels,
    preds,
  )


def pr_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> N:
  """Compute PR."""
  precision, recall, _ = filter_pure_callback(
    m.precision_recall_curve,
    labels,
    preds,
    result_shape_dtypes=(
      jax.ShapeDtypeStruct((preds.shape[0] + 1,), jnp.float32),
      jax.ShapeDtypeStruct((preds.shape[0] + 1,), jnp.float32),
      jax.ShapeDtypeStruct((preds.shape[0],), jnp.float32),
    ),
  )
  return jax.pure_callback(
    m.auc, jax.ShapeDtypeStruct((), jnp.float32), recall, precision
  )


class AbstractMetric[**P](Module):
  """Abstract Metric."""

  name: AbstractVar[str | None]

  @abstractmethod
  def __or__(self, value: Self) -> Self:
    """Or."""
    raise NotImplementedError

  @abstractmethod
  def __add__(self, value: Self) -> Self:
    """Add."""
    raise NotImplementedError

  def best(self, *args: P.args, **kwgs: P.kwargs) -> Self:
    """Best value."""
    return self[self.best_idx(*args, **kwgs)]

  def best_idx(self, *args: P.args, **kwgs: P.kwargs) -> Int[Array, "1"]:
    """Best value."""
    raise NotImplementedError

  def __repr__(self) -> str:
    """Representation."""
    return tree_pformat(self, short_arrays=False)

  @abstractmethod
  def __getitem__(self, idx: int | Int[Array, "#b"] | slice) -> Self:
    """Get item."""
    raise NotImplementedError


class Metric(AbstractMetric):
  """Base Metric."""

  value: N = field(default=jnp.nan, converter=convert)
  order: Literal["max", "min"] = "max"
  name: str | None = None

  def __or__(self, value: Self) -> Self:
    """Or."""
    s1, _ = partition(self, lambda x: is_array(x) and (not jnp.isnan(x).any()))
    return combine(s1, value)

  def __add__(self, value: Self) -> Self:
    """Add."""
    if self.name != value.name:
      raise ValueError(f"Name not match: this {self.name=} != {value.name=}")
    if self.order != value.order:
      raise ValueError(f"Order not match: this {self.order=} != {value.order=}")
    return filter_concat([self, value])

  def _max(self) -> Int[Array, "1"]:
    """Best value."""
    return jnp.nanargmax(self.value)

  def _min(self) -> Int[Array, "1"]:
    """Worst value."""
    return jnp.nanargmin(self.value)

  def best_idx(self) -> Int[Array, "1"]:
    """Best value."""
    return self._max() if self.order == "max" else self._min()

  def __repr__(self) -> str:
    """Representation."""
    return tree_pformat(self, short_arrays=False)

  def __getitem__(self, idx: int | Int[Array, "#b"] | slice) -> Self:
    """Get item."""
    if self.value is None:
      raise ValueError("No value.")
    return tree_at(lambda x: x.value, self, self.value[idx])


class Accuracy(Metric):
  """Accuracy."""

  @classmethod
  def create(
    cls,
    target: Num[Array, "*b a"],
    pred: Num[Array, "*b a"],
    name: str = "accuracy",
  ) -> Self:
    """Create from data."""
    if target.shape != pred.shape:
      raise ValueError(f"Shape not match: {target.shape=} != {pred.shape=}")
    match target.ndim:
      case 1:
        return cls(value=jnp.mean(target == pred), name=name)
      case 2:
        return cls(value=jnp.mean(target == pred, axis=1), name=name)
      case _:
        raise ValueError(f"Unsupported shape: {target.shape=}")

  @classmethod
  def create_empty(cls, name: str = "accuracy") -> Self:
    """Create empty."""
    return cls(
      value=jnp.nan,
      name=name
    )


class GroupMetric(AbstractMetric):
  """Group Metric."""

  metrics: list[AbstractMetric]
  name: str | None = None

  def __or__(self, value: Self) -> Self:
    """Or."""
    if tree_equal(*jax.tree.map(jnp.shape, [self.metrics, value.metrics])):
      s1, _ = partition(self, lambda x: is_array(x) and (not jnp.isnan(x).all()))
      return combine(s1, value)
    raise ValueError("Shape not match.")

  def __add__(self, value: Self) -> Self:
    """Add."""
    if self.name != value.name:
      raise ValueError(f"Name not match: this {self.name=} != {value.name=}")
    return filter_concat([self, value])

  def __getitem__(self, idx: int | Array | slice) -> Self:
    """Get item."""
    return tree_at(
      lambda x: x.metrics, self, jax.tree.map(operator.itemgetter(idx), self.metrics)
    )

  def best_idx(
    self, which: int | Callable[[list[AbstractMetric]], Metric]
  ) -> Int[Array, "1"]:
    """Best value."""
    match which:
      case Callable():
        return which(self.metrics).best_idx()
      case int():
        return self.metrics[which].best_idx()
      case _:
        raise TypeError("which should be int or list[m] -> m.")


class AccRocPR(GroupMetric):
  """Accuracy, ROC, PR."""

  @classmethod
  def create(
    cls,
    target: Num[Array, "*b a"],
    pred: Num[Array, "*b a"],
    name: str = "accrocpr",
  ) -> Self:
    """Create from data."""
    if target.shape != pred.shape:
      raise ValueError(f"Shape not match: {target.shape=} != {pred.shape=}")
    match target.ndim:
      case 1:
        return cls(
          metrics=[
            Metric(value=jnp.mean(target == pred), name="acc"),
            Metric(value=roc_auc_score(target, pred), name="roc"),
            Metric(value=pr_auc_score(target, pred), name="pr"),
            Metric(value=average_precision_score(target, pred), name="ap"),
          ],
          name=name,
        )
      case 2:
        return cls(
          metrics=[
            Metric(value=jnp.mean(target == pred, axis=1), name="acc"),
            Metric(value=jax.vmap(roc_auc_score)(target, pred), name="roc"),
            Metric(value=filter_vmap(pr_auc_score)(target, pred), name="pr"),
            Metric(value=filter_vmap(average_precision_score)(target, pred), name="ap"),
          ],
          name=name,
        )
      case _:
        raise ValueError(f"Unsupported shape: {target.shape=}")

  @classmethod
  def create_empty(cls, name: str = "accrocpr") -> Self:
    """Create empty."""
    return cls(
      metrics=[
        Metric(value=jnp.nan, name="acc"),
        Metric(value=jnp.nan, name="roc"),
        Metric(value=jnp.nan, name="pr"),
        Metric(value=jnp.nan, name="ap"),
      ],
      name=name
    )


def dice_coef(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute dice coefficient."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred)

  return (2.0 * intersection) / (union + eps)


def iou_coef(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute intersection over union."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred) - intersection

  return intersection / (union + eps)
