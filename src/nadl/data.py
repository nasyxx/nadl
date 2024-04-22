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
date     : Apr 19, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : data.py
project  : nadl
license  : GPL-3.0+

Simple dataset and dataloader.
"""

from collections.abc import Iterator
import equinox as eqx
import jax
import jax.numpy as jnp
from .keys import Keys
from typing import NamedTuple


class DState(NamedTuple):
  """Dataloader state."""

  idx: jax.Array
  pad: int | None = None


class IdxDataloader(eqx.Module):
  """Simple index dataloader.

  Provide indexes only for datasets.
  """

  length: int
  key: Keys
  batch_size: int = 1
  shuffle: bool = False
  drop_last: bool = False
  auto_pad: bool = True

  def __init__(
    self,
    length: int,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    auto_pad: bool = False,
    key: jax.Array | None = None,
  ) -> None:
    """Initiate the dataloader."""
    self.length = length
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.auto_pad = auto_pad
    self.key = Keys.from_int_or_key(key or 42)

  def __call__(self, ki: int) -> Iterator[DState]:
    """Get the indexes."""
    idxes = (
      jax.random.permutation(self.key(ki)[1], self.length)
      if self.shuffle
      else jnp.arange(self.length)
    )
    if self.drop_last:
      length = self.length - self.length % self.batch_size
      idxes = idxes[:length]
      pad = 0

    pad = self.batch_size - len(idxes) % self.batch_size
    if 0 < pad < self.batch_size:
      idxes = jnp.r_[idxes, jnp.zeros(pad, idxes.dtype)]
    idxes = idxes.reshape(-1, self.batch_size)

    for i, idx in enumerate(idxes):
      if pad and i >= idxes.shape[0] - 1:
        if self.auto_pad:
          yield DState(idx, pad)
        else:
          yield DState(idx[:-pad], 0)
      else:
        yield DState(idx, 0)


def __test() -> None:
  """Test."""
  dl = IdxDataloader(10, 3, shuffle=True, drop_last=True, auto_pad=True)
  for i, state in enumerate(dl(42)):
    print(i, state.idx, state.pad)


if __name__ == "__main__":
  __test()
