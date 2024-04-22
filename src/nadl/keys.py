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

from collections import deque
from collections.abc import Sequence

import jax
from equinox import Module, field


class Keys(Module):
  """JAX random key sequence."""

  key: jax.Array
  subkeys: deque[jax.Array]
  init_key: jax.Array
  step: jax.Array = field(default_factory=lambda: jax.numpy.array(0))

  @property
  def state(self) -> tuple[jax.Array, Sequence[jax.Array], jax.Array]:
    """Get state."""
    return self.key, self.subkeys, self.step

  @classmethod
  def from_int_or_key(cls: type["Keys"], key: jax.Array | int) -> "Keys":
    """Convert int or key to Keys."""
    if isinstance(key, int):
      return cls(key := jax.random.key(key), deque(), key)
    return cls(key, deque(), key)

  @classmethod
  def from_state(
    cls: type["Keys"], key: jax.Array, subkeys: Sequence[jax.Array], step: jax.Array
  ) -> "Keys":
    """Convert state to Keys."""
    return cls(key, deque(subkeys), step)

  def reverse(self, num: int) -> "Keys":
    """Reverse the keys."""
    while (num := num - 1) >= 0:
      if self.subkeys:
        self.subkeys.append(jax.random.fold_in(self.subkeys[-1], 0))
      else:
        self.subkeys.append(jax.random.fold_in(self.key, 0))
    return self

  def next_key(self) -> tuple["Keys", jax.Array]:
    """Get next key."""
    if not self.subkeys:
      self.reverse(1)
    key = self.subkeys.popleft()
    return self.from_state(key, self.subkeys, self.step + 1), key

  def take(self, num: int) -> tuple["Keys", tuple[jax.Array, ...]]:
    """Take num keys."""
    self.reverse(max(num - len(self.subkeys), 0))
    keys = tuple(self.subkeys.popleft() for _ in range(num))
    return self.from_state(keys[-1], self.subkeys, self.step + num), keys

  def __call__(self, epoch: int | None = None) -> tuple["Keys", jax.Array]:
    """Get keys for epoch."""
    if epoch:
      if epoch > self.step:
        keys, ks = self.take((epoch - self.step).item())
        return keys, ks[-1]
      key = Keys.from_int_or_key(self.init_key)
      keys, ks = key.take(epoch)
      return keys, ks[-1]
    return self.next_key()


new_key = Keys.from_int_or_key
