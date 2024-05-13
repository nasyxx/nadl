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

from collections.abc import Callable, Iterator
import equinox as eqx
import jax
import jax.numpy as jnp
from .keys import Keys
from typing import NamedTuple


class DState[T](NamedTuple):
  """Dataloader state."""

  xs: T
  pad: int | None = None


class IdxDataloader[T](eqx.Module):
  """Simple index dataloader.

  Provide indexes only for datasets.
  """

  length: int
  key: Keys
  batch_size: int = 1
  shuffle: bool = False
  drop_last: bool = False
  auto_pad: bool = True
  transform: Callable[[jax.Array], T] | None = None

  def __init__(
    self,
    length: int,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    auto_pad: bool = False,
    key: jax.Array | None = None,
    transform: Callable[[jax.Array], T] | None = None,
  ) -> None:
    """Initiate the dataloader."""
    self.length = length
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.auto_pad = auto_pad
    self.key = Keys.from_int_or_key(key or 42)
    self.transform = transform

  def __call__(self, ki: int) -> Iterator[DState[T]]:
    """Get the indexes."""
    idxes = (
      jax.random.permutation(self.key(ki)[1], self.length)
      if self.shuffle
      else jnp.arange(self.length)
    )
    length = (
      self.length if not self.drop_last else self.length - self.length % self.batch_size
    )

    pad = (
      (self.batch_size - r) % self.batch_size if (r := length % self.batch_size) else 0
    )
    pad = pad if pad != self.batch_size else 0

    if pad:
      idxes = jnp.r_[idxes, jnp.full(pad, -1, idxes.dtype)]  # padding placeholder

    idxes = idxes[:length + pad].reshape(-1, self.batch_size)

    for i in idxes:
      ii = i
      if pad and not self.auto_pad and (ii == -1).any():
        ii = ii[ii != -1]
      xs = self.transform(ii) if self.transform else ii
      yield DState(xs, pad if ((ii == -1).any() and self.auto_pad) else None)


def __test() -> None:
  """Test."""
  dl = IdxDataloader(10, 3, shuffle=True, drop_last=False, auto_pad=True)
  for i, state in enumerate(dl(42)):
    print(i, state.xs, state.pad)
  dl = IdxDataloader(10, 3, shuffle=True, drop_last=False, auto_pad=False)
  for i, state in enumerate(dl(42)):
    print(i, state.xs, state.pad)
  dl = IdxDataloader(10, 3, shuffle=False, drop_last=True, auto_pad=False)
  for i, state in enumerate(dl(42)):
    print(i, state.xs, state.pad)


if __name__ == "__main__":
  __test()
