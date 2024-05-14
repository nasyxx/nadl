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
from equinox import tree_at

from .keys import Keys
from .loops import PG


class DState[T](eqx.Module):
  """Dataloader state."""

  xs: T
  pad: int | None = None
  epoch: int | None = None
  step: int | None = None
  name: str | None = None


class IdxDataloader[T](eqx.Module):
  """Simple index dataloader.

  Provide indexes only dataloader for dataset.
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

    # use -1 as padding placeholder
    idxes = jnp.r_[idxes, jnp.full(pad, -1, idxes.dtype)]

    idxes = idxes[: length + pad].reshape(-1, self.batch_size)

    for i in idxes:
      ii = i
      if pad and not self.auto_pad and (ii == -1).any():
        ii = ii[ii != -1]
      xs = self.transform(ii) if self.transform else ii
      yield DState(xs, pad if ((ii == -1).any() and self.auto_pad) else None)

  def __len__(self) -> int:
    """Length."""
    if self.drop_last:
      return self.length // self.batch_size
    return (self.length + self.batch_size - 1) // self.batch_size


def es_loop[T](
  loader: IdxDataloader[T],
  pg: PG,
  epochs: int = 2,
  start_epoch: int = 1,
  prefix: str = "L",
  es: str = "E",
  ss: str = "S",
) -> Iterator[DState[T]]:
  """Simple epoch loop."""
  es, ss = f"{prefix}-{es}", f"{prefix}-{ss}"
  if epochs > 1:
    if es in pg.tasks:
      pg.pg.reset(pg.tasks[es])
    else:
      pg.add_task(es, total=epochs, res="")
    pg.advance(pg.tasks[es], start_epoch - 1)

  if ss in pg.tasks:
    pg.pg.reset(pg.tasks[ss])
  else:
    pg.add_task(ss, total=len(loader) * epochs, res="")
  pg.advance(pg.tasks[ss], (start_step := max((start_epoch - 1), 0) * len(loader)))

  for i in range(start_epoch, epochs + 1):
    for ii, ds in enumerate(loader(i), start_step):
      yield tree_at(
        lambda x: (x.epoch, x.step, x.name),
        ds,
        (i, (i - 1) * len(loader) + ii + 1, prefix),
        is_leaf=lambda x: x is None,
      )
      pg.advance(pg.tasks[ss])
    if epochs > 1:
      pg.advance(pg.tasks[es])


def __test() -> None:
  """Test."""
  pg = PG.init_progress()
  with pg:
    pg.console.print("Drop Last: False, Auto Pad: True")
    dl = IdxDataloader(10, 3, shuffle=True, drop_last=False, auto_pad=True)
    for i in es_loop(dl, pg, prefix="DFAT"):
      pg.update_res("DFAT-S", {"epoch": i.epoch, "step": i.step, "name": i.name})
      pg.console.print(i, i.xs)
    pg.console.print("Drop Last: False, Auto Pad: False")
    dl = IdxDataloader(10, 3, shuffle=True, drop_last=False, auto_pad=False)
    for i in es_loop(dl, pg, prefix="DFAF"):
      pg.console.print(i, i.xs)
    pg.console.print("Drop Last: True, Auto Pad: False")
    dl = IdxDataloader(10, 3, shuffle=False, drop_last=True, auto_pad=False)
    for i in es_loop(dl, pg, prefix="DTAF"):
      pg.console.print(i, i.xs)
    pg.console.print("Drop Last: True, Auto Pad: True")
    dl = IdxDataloader(10, 3, shuffle=False, drop_last=True, auto_pad=True)
    for i in es_loop(dl, pg, prefix="DTAT"):
      pg.console.print(i, i.xs)


if __name__ == "__main__":
  __test()
