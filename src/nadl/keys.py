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

from typing import NamedTuple


class Keys(NamedTuple):
  """JAX random key sequence."""

  key: jax.Array
  subkeys: deque[jax.Array]


def from_int_or_key(key: jax.Array | int) -> Keys:
  """Convert int or key to Keys."""
  if isinstance(key, int):
    return Keys(jax.random.key(key), deque())
  return Keys(key, deque())


new_key = from_int_or_key


def from_state(key: jax.Array, subkeys: Sequence[jax.Array]) -> Keys:
  """Convert state to Keys."""
  return Keys(key, deque(subkeys))


def reverse(keys: Keys, num: int) -> Keys:
  """Reverse the keys."""
  key, subkeys = keys
  while (num := num - 1) >= 0:
    if subkeys:
      subkeys.append(jax.random.fold_in(subkeys[-1], 0))
    else:
      subkeys.append(jax.random.fold_in(key, 0))
  return keys._replace(subkeys=subkeys)


def next_key(keys: Keys) -> Keys:
  """Get next key."""
  if not keys.subkeys:
    keys = reverse(keys, 1)
  return from_state(keys.subkeys.popleft(), keys.subkeys)


def take(keys: Keys, num: int) -> tuple[Keys, ...]:
  """Take num keys."""
  keys = reverse(keys, max(num - len(keys.subkeys), 0))
  return tuple(map(lambda _: next_key(keys), range(num)))
