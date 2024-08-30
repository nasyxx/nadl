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
filename : keys.py
project  : nadl
license  : GPL-3.0+

Keys.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from equinox import Module, tree_at

from jaxtyping import ArrayLike, Int, PRNGKeyArray, ScalarLike

# ruff: noqa: F722


class Keys(Module):
  """JAX random key sequence.

  init_key:  The initial key.
  keys: The key sequence in cluding the histories.
  idx: Current key idx in the key sequence.
  """

  init_key: PRNGKeyArray
  keys: PRNGKeyArray
  idx: ScalarLike = 0

  def __call__(self, epoch: ArrayLike) -> jax.Array:
    """Get keys for epoch."""
    return jax.random.fold_in(self.init_key, epoch)

  def __iter__(self) -> Keys:
    """Iterate the keys."""
    return self

  def __next__(self) -> Keys:
    """Get next key."""
    return self.next_key()

  def __len__(self) -> int:
    """Length of keys."""
    return self.keys.shape[0]

  @property
  def key(self) -> PRNGKeyArray:
    """Get key."""
    if self.keys.shape[0]:
      return self.keys[-1]
    return self(self.idx + 1)

  @property
  def state(self) -> tuple[PRNGKeyArray, PRNGKeyArray, ScalarLike]:
    """Get state."""
    return self.init_key, self.keys, self.idx

  @classmethod
  def from_int_or_key(cls: type[Keys], key: PRNGKeyArray | int) -> Keys:
    """Convert int or key to Keys."""
    if isinstance(key, int):
      key = jax.random.key(key)
    return cls(key, jax.random.fold_in(key, 0).reshape(-1), jnp.asarray(0))

  @classmethod
  def from_state(
    cls: type[Keys], key: PRNGKeyArray, keys: PRNGKeyArray, idx: ScalarLike
  ) -> Keys:
    """Convert state to Keys."""
    return cls(key, keys, idx)

  def reserve(self, num: int | jax.Array) -> Keys:
    """Reverse the keys."""
    return tree_at(
      lambda x: x.keys,
      self,
      jax.vmap(self)(jnp.arange(jnp.maximum(num, len(self)))),
    )

  def next_key(self) -> Keys:
    """Get next key."""
    return tree_at(lambda k: k.idx, self, self.idx + 1)

  def take(self, num: int) -> PRNGKeyArray:
    """Take num keys."""
    return jax.vmap(self)(jnp.arange(num)) if num > len(self) else self.keys[-num:]


new_key = Keys.from_int_or_key
