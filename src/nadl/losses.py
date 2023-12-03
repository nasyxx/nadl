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
date     : Nov 30, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : losses.py
project  : nadl
license  : GPL-3.0+

Extra Loss functions.
  - focal loss
  - dice loss
"""
from typing import Literal
import jax
import jax.numpy as jnp
from optax import sigmoid_binary_cross_entropy
from jax.nn import log_softmax, log_sigmoid, sigmoid, softmax
from .utils import classit
from .metrics import dice_coef


def sigmoid_focal_loss(
  logits: jax.Array, labels: jax.Array, alpha: float = 0.25, gamma: float = 2.0
) -> jax.Array:
  """Focal loss."""
  p = sigmoid(logits)
  ce_loss = sigmoid_binary_cross_entropy(logits, labels)
  p_t = p * labels + (1 - p) * (1 - labels)
  loss = ce_loss * ((1 - p_t) ** gamma)

  alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
  return alpha_t * loss


def softmax_focal_loss(
  logits: jax.Array, labels: jax.Array, alpha: float = 0.25, gamma: float = 2.0
) -> jax.Array:
  """Focal loss."""
  focus = jnp.power(-jax.nn.softmax(logits, axis=-1) + 1.0, gamma)
  loss = -labels * alpha * focus * jax.nn.log_softmax(logits, axis=-1)
  return jnp.sum(loss, axis=-1)


def dice_loss(
  logits: jax.Array,
  labels: jax.Array,
  method: Literal[None, "sigmoid", "softmax"] = "sigmoid",
) -> jax.Array:
  """Dice loss."""
  if labels.ndim == 1:
    labels = labels[:, None]
  if logits.ndim == 1:
    logits = classit(logits[:, None], method=method)
  return jax.vmap(dice_coef)(labels, logits)
