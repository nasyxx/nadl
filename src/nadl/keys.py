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
from equinox import Module, field, tree_at

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Sequence


class Keys(Module):
  """JAX random key sequence.

  init_key:  The initial key.
  keys: The key sequence in cluding the histories.
  idx: Current key idx in the key sequence.
  """

  init_key: jax.Array
  keys: list[jax.Array]
  idx: jax.Array = field(default_factory=lambda: jax.numpy.array(0))

  @property
  def key(self) -> jax.Array:
    """Get key."""
    if self.keys:
      return self.keys[-1]
    self.reserve(1)
    return self.keys[-1]

  @property
  def state(self) -> tuple[jax.Array, Sequence[jax.Array], jax.Array]:
    """Get state."""
    return self.init_key, self.keys, self.idx

  @classmethod
  def from_int_or_key(cls: type[Keys], key: jax.Array | int) -> Keys:
    """Convert int or key to Keys."""
    if isinstance(key, int):
      return cls(jax.random.key(key), [])
    return cls(key, [])

  @classmethod
  def from_state(
    cls: type[Keys], key: jax.Array, keys: Sequence[jax.Array], idx: jax.Array
  ) -> Keys:
    """Convert state to Keys."""
    return cls(key, list(keys), idx)

  def reserve(self, num: int | jax.Array) -> Keys:
    """Reverse the keys."""
    while (num := num - 1) >= 0:
      if self.keys:
        self.keys.append(jax.random.fold_in(self.keys[-1], 0))
      else:
        self.keys.append(jax.random.fold_in(self.init_key, 0))
    return self

  def next_key(self) -> Keys:
    """Get next key."""
    return tree_at(lambda k: k.idx, self, self.idx + 1)

  def __iter__(self) -> Keys:
    """Iterate the keys."""
    return self

  def __next__(self) -> Keys:
    """Get next key."""
    return self.next_key()

  def take(self, num: int) -> jax.Array:
    """Take num keys."""
    self.reserve(jnp.max(num - len(self.keys), 0))
    return jnp.r_[*self.keys[-num:]]

  def __len__(self) -> int:
    """Length of keys."""
    return len(self.keys)

  def __call__(self, epoch: int | jax.Array | None = None) -> jax.Array:
    """Get keys for epoch."""
    match epoch:
      case int():
        if epoch + 1 > len(self.keys):
          self.reserve(epoch + 1 - len(self.keys))
        return self.keys[epoch]
      case jax.Array():
        self.reserve(jnp.maximum((epoch + 1 - len(self.keys)).max(), 0))
        return jnp.r_[*self.keys][epoch]
      case None:
        return self(self.idx)


new_key = Keys.from_int_or_key
