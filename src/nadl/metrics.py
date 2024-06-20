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
from warnings import catch_warnings, simplefilter, warn

import jax
import jax.numpy as jnp
from equinox import (
  AbstractVar,
  Module,
  combine,
  filter_pure_callback,
  is_array,
  partition,
  tree_equal,
  tree_pformat,
)
from optax import softmax_cross_entropy_with_integer_labels

from jaxtyping import Array, ArrayLike, Float, Int, Num, Scalar
from typing import Literal, Self, cast

import numpy as np
import sklearn.metrics as m

from .utils import batch_array_p, filter_concat, filter_tree


def convert(x: Num[ArrayLike, "..."]) -> Num[Array, "B ..."]:
  """Convert to float."""
  x = jnp.asarray(x)
  return x.reshape(-1, *x.shape[1:])


def roc_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> Scalar:
  """Compute ROC."""
  return jax.pure_callback(
    m.roc_auc_score, jax.ShapeDtypeStruct((), jnp.float32), labels, preds
  )


def average_precision_score(
  labels: Num[Array, " A"],
  preds: Num[Array, " A"],
  average: Literal["micro", "macro"] = "macro",
) -> Scalar:
  """Compute PR."""
  return jax.pure_callback(
    partial(m.average_precision_score, average=average),
    jax.ShapeDtypeStruct((), jnp.float32),
    labels,
    preds,
  )


def pr_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> Scalar:
  """Compute PR."""

  def _callback(lbl: Num[Array, " A"], prd: Num[Array, " A"]) -> Scalar:
    precision, recall, _ = m.precision_recall_curve(lbl, prd)
    return jnp.asarray(m.auc(recall, precision))

  return filter_pure_callback(
    _callback,
    labels,
    preds,
    result_shape_dtypes=jax.ShapeDtypeStruct((), jnp.float32),
  )


class Metric[**P, T](Module):
  """Base Metric.

  This metric is for batch data not for a single one.
  Thus, please consider the shape of the input data as [b, ...].

  The default getitem will only take the array with ndim > 1.
  """

  name: AbstractVar[str | None]

  def __post_init__(self) -> None:
    """Post init."""
    for k, v in self.__dict__.items():
      if isinstance(v, Array | np.ndarray):
        self.__dict__[k] = convert(v)

  def __check_init__(self) -> None:  # noqa: PLW3201
    """Check init."""
    arrs = filter_tree(self, batch_array_p)
    if arrs:
      if not tree_equal(*jax.tree.map(lambda x: jnp.shape(x)[0], arrs)):
        raise ValueError("All batched array should have the same batch size.")
    else:
      warn("No batched array found in the metric.", stacklevel=1)

  @classmethod
  def merge(cls, *metrics: Self) -> Self:
    """Merge all metrics."""
    _, s2 = partition(metrics, is_array)
    if not tree_equal(*s2):
      raise ValueError("All metrics should have the same non-array values.")
    return filter_concat(metrics, batch_array_p)

  def __add__(self, value: Self) -> Self:
    """Add."""
    return self.merge(self, value)

  def __or__(self, value: Self) -> Self:
    """Or."""
    return combine(self, value)

  @classmethod
  def empty(cls, *_args: P.args, **_kwds: P.kwargs) -> Self:
    """Empty."""
    with catch_warnings():
      simplefilter("ignore")
      return cls(**dict.fromkeys(cls.__dataclass_fields__))

  def compute(self) -> T:
    """Compute."""
    raise NotImplementedError

  def __getitem__(self, idx: int | slice | Int[ArrayLike, "..."]) -> Self:
    """Get item.

    The default getitem will only take the array with ndim > 1.
    """
    return jax.tree.map(lambda x: x[idx] if batch_array_p(x) else x, self)

  def show(self) -> str:
    """Show."""
    return tree_pformat(self, short_arrays=False)


class Accuracy(Metric):
  """Accuracy."""

  labels: Int[Array, "..."]
  preds: Int[Array, "..."]
  name: str = "accuracy"

  def compute(self) -> Array:
    """Compute."""
    return jnp.nanmean(self.labels == self.preds, axis=-1)


def dice_coef(
  y_true: Int[Array, " A"], y_pred: Int[Array, " A"], eps: float = 1e-8
) -> Scalar:
  """Compute dice coefficient."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred)

  return (2.0 * intersection) / (union + eps)


def iou_coef(
  y_true: Int[Array, " A"], y_pred: Int[Array, " A"], eps: float = 1e-8
) -> Scalar:
  """Compute intersection over union."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred) - intersection

  return intersection / (union + eps)


def info_nce(
  pos: Float[Array, "B D"], neg: Float[Array, "B D"], t: float = 0.07
) -> Float[Array, "B 1"]:
  """Compute info nce loss for paired pos-neg data."""
  pos /= jnp.linalg.norm(pos, axis=-1, keepdims=True)
  neg /= jnp.linalg.norm(neg, axis=-1, keepdims=True)

  pos_sim = jnp.einsum("bi,bi->b", pos, pos) / t
  neg_sim = jnp.einsum("bi,nd->bn", pos, neg) / t
  logits = jnp.c_[pos_sim, neg_sim]
  return cast(
    Float[Array, "B 1"],
    softmax_cross_entropy_with_integer_labels(logits, jnp.arange(logits.shape[0])),
  )


def _test() -> None:
  """Test."""
  k1, k2 = jax.random.split(jax.random.key(42))
  p = jax.random.normal(k1, (4, 30))
  n = jax.random.normal(k2, (4, 30))
  assert info_nce(p, n).shape == (4,)


if __name__ == "__main__":
  _test()
