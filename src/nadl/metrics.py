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

Some simple metrics.  For complex metrics, use sklearn.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("normalize",))
def accuracy(y_true: jax.Array, y_pred: jax.Array, normalize: bool = True) -> jax.Array:
  """Accuracy."""
  return jnp.mean(y_true == y_pred) if normalize else jnp.sum(y_true == y_pred)


def roc_curve(
  y_true: jax.Array, y_score: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute Receiver operating characteristic (ROC).  Cannot be jitted."""
  y_true = jnp.asarray(y_true)
  y_score = jnp.asarray(y_score)

  descending_order = jnp.argsort(y_score)[::-1]
  y_score = y_score[descending_order]
  y_true = y_true[descending_order]

  # TPR and FPR
  distinct_value_indices = jnp.argwhere(jnp.diff(y_score))
  threshold_idxs = jnp.r_[distinct_value_indices, y_true.size - 1]

  tps = jnp.cumsum(y_true)[threshold_idxs]
  fps = 1 + threshold_idxs - tps

  tpr = tps / jnp.sum(y_true)
  fpr = fps / jnp.sum(1 - y_true)

  thresholds = jnp.r_[jnp.inf, y_score[threshold_idxs]]

  tpr = jnp.r_[0, tpr]
  fpr = jnp.r_[0, fpr]

  return fpr, tpr, thresholds


def precision_recall_curve(
  y_true: jax.Array, y_score: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compute precision-recall pairs for different probability thresholds."""
  y_true = jnp.asarray(y_true)
  y_score = jnp.asarray(y_score)

  descending_order = jnp.argsort(y_score)[::-1]
  y_score = y_score[descending_order]
  y_true = y_true[descending_order]

  # Precision and recall
  distinct_value_indices = jnp.argwhere(jnp.diff(y_score))
  threshold_idxs = jnp.r_[distinct_value_indices, y_true.size - 1]

  tps = jnp.cumsum(y_true)[threshold_idxs]
  fps = 1 + threshold_idxs - tps

  precision = tps / (tps + fps)
  recall = tps / jnp.sum(y_true)

  thresholds = jnp.r_[jnp.inf, y_score[threshold_idxs]]

  precision = jnp.r_[1, precision]
  recall = jnp.r_[0, recall]

  return precision, recall, thresholds


@jax.jit
def auc(fpr: jax.Array, tpr: jax.Array) -> jax.Array:
  """Compute Area Under the Curve (AUC) from the ROC curve."""
  # Sort by false positive rates
  sorted_indices = jnp.argsort(fpr)
  fpr, tpr = fpr[sorted_indices], tpr[sorted_indices]

  # Compute the area using the trapezoidal rule
  return jax.scipy.integrate.trapezoid(tpr, fpr)


@partial(jax.jit, static_argnames=("num_classes",))
def confusion_matrix(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2
) -> jax.Array:
  """Compute confusion matrix."""
  return (
    jnp.zeros((num_classes, num_classes), dtype=jnp.int32).at[(y_true, y_pred)].add(1)
  )


@partial(jax.jit, static_argnames=("num_classes", "eps"))
def precision_recall_fscore_support(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2, eps: float = 1e-8
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Compute precision, recall, F-score and support for each class."""
  # Create a confusion matrix
  conf_matrix = confusion_matrix(y_true, y_pred, num_classes)

  # Compute precision, recall for each class
  true_positives = jnp.diag(conf_matrix)
  predicted_positives = jnp.sum(conf_matrix, axis=0)
  actual_positives = jnp.sum(conf_matrix, axis=1)

  precision = true_positives / (predicted_positives + eps)
  recall = true_positives / (actual_positives + eps)

  # Compute F1 score
  f1_score = 2 * (precision * recall) / (precision + recall + eps)

  # Compute support for each class
  support = actual_positives

  return precision, recall, f1_score, support


@partial(jax.jit, static_argnames=("num_classes", "eps"))
def precision_score(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2, eps: float = 1e-8
) -> jax.Array:
  """Compute the precision."""
  precision, _, _, _ = precision_recall_fscore_support(y_true, y_pred, num_classes, eps)
  return precision


@partial(jax.jit, static_argnames=("num_classes", "eps"))
def recall_scroe(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2, eps: float = 1e-8
) -> jax.Array:
  """Compute the recall."""
  _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, num_classes, eps)
  return recall


@partial(jax.jit, static_argnames=("num_classes", "eps"))
def f1_score(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2, eps: float = 1e-8
) -> jax.Array:
  """Compute the F1 score."""
  _, _, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, num_classes, eps)
  return f1_score


@partial(jax.jit, static_argnames=("num_classes", "eps"))
def support(
  y_true: jax.Array, y_pred: jax.Array, num_classes: int = 2, eps: float = 1e-8
) -> jax.Array:
  """Compute the support."""
  _, _, _, support = precision_recall_fscore_support(y_true, y_pred, num_classes, eps)
  return support


def roc_auc_score(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
  """Compute Area Under the ROC of AUC from prediction scores."""
  # Compute ROC curve
  fpr, tpr, _ = roc_curve(y_true, y_score)
  return auc(fpr, tpr)


def average_precision_score(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
  """Compute average precision from prediction scores."""
  # Compute precision-recall curve
  precision, recall, _ = precision_recall_curve(y_true, y_score)
  return jax.scipy.integrate.trapezoid(precision, recall)


def dice_coef(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute dice coefficient."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred)

  return (2.0 * intersection) / (union + eps)


def iou(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute intersection over union."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred) - intersection

  return intersection / (union + eps)
