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

from __future__ import annotations

import jax
import jax.numpy as jnp
from equinox import Module, filter_jit, filter_vmap

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .keys import Keys
from .loops import PG, RESC, PGThread

if TYPE_CHECKING:
  from collections.abc import Callable, Iterator

  from jaxtyping import Array, Bool, Int


class DState[T](NamedTuple):
  """Dataloader state."""

  xs: T
  pad: Bool[Array, " b"]
  epoch: Int[Array, ""] = jnp.asarray(0)
  step: Int[Array, ""] = jnp.asarray(0)
  name: str | None = None


def _np_sort(x: jax.Array, axis: int | None = None) -> np.ndarray:
  """Sort the array."""
  return np.argsort(np.asarray(x), axis=axis)


@filter_jit
def fallback_argsort(x: jax.Array, axis: int | None = None) -> jax.Array:
  """Fallback to numpy argsort when CPU."""
  if jax.devices()[0].platform == "cpu":
    return jax.pure_callback(
      _np_sort, jax.ShapeDtypeStruct(x.shape, jnp.int32), x, axis
    )
  return x.argsort(axis=axis)


class IdxDataloader(Module):
  """Simple index dataloader."""

  length: int
  pad: int
  batch_size: int
  drop_num: int

  def __init__(
    self,
    length: int,
    batch_size: int,
    drop_last: bool = False,
  ) -> None:
    """Initiate the dataloader."""
    self.length = length
    length = length if not drop_last else length - length % batch_size
    pad = (batch_size - r) % batch_size if (r := length % batch_size) else 0
    self.pad = pad if pad != batch_size else 0
    self.batch_size = batch_size
    self.drop_num = self.length % batch_size if drop_last else 0

  def __call__(self, key: jax.Array | None = None) -> DState[Int[Array, " b d"]]:
    """Get the indexes."""
    idxes = jnp.arange(self.length)
    if key is not None:
      idxes = jnp.take_along_axis(
        idxes,
        # NOTE: Fallback to numpy argsort since it has performance isssue in CPU.
        # https://github.com/google/jax/issues/10434
        fallback_argsort(jax.random.uniform(key, (self.length,))),
        axis=0,
      )
    length = self.length if not self.drop_num else self.length - self.drop_num

    idxes = jnp.r_[idxes, jnp.full(self.pad, -1, idxes.dtype)]
    idxes = idxes[: length + self.pad].reshape(-1, self.batch_size)
    return DState(idxes, jnp.where(idxes == -1, 1, 0).astype(bool))

  def __len__(self) -> int:
    """Length of the dataloader."""
    return self.length // self.batch_size + (
      1 if (not self.drop_num) and self.length % self.batch_size else 0
    )


def es_loop[T](
  dl: IdxDataloader,
  pg: PG,
  keys: Keys | None = None,
  epochs: int = 1,
  start_epoch: int = 1,
  prefix: str = "L",
  es: str = "E",
  ss: str = "S",
  transform: Callable[[Int[Array, " a"]], T] | None = None,
) -> Iterator[DState[T]]:
  """Simple epoch loop."""
  assert epochs > 0, "Epochs should be greater than 0."
  if transform:
    transform = filter_jit(transform)

  steps = len(dl)
  es, ss = f"{prefix}-{es}", f"{prefix}-{ss}"
  if es in pg.tasks:
    pg.pg.reset(pg.tasks[es], total=epochs)
  else:
    pg.add_task(es, total=epochs, res="", visible=epochs > 1)
  pg.advance(pg.tasks[es], start_epoch - 1)
  if ss in pg.tasks:
    pg.pg.reset(pg.tasks[ss], total=steps * epochs, res="")
  else:
    pg.add_task(ss, total=steps * epochs, res="")
  pg.advance(pg.tasks[ss], (start_epoch - 1) * steps)

  kf = filter_jit(filter_vmap(keys.reserve(epochs))) if keys else lambda _: None
  df = filter_jit(filter_vmap(dl, axis_size=1))
  ds = df() if keys is None else df(kf(jnp.arange(epochs)))

  @jax.jit
  def _form(i: jax.Array, ii: jax.Array) -> tuple[jax.Array, jax.Array]:
    return i + 1, i * steps + ii + 1

  with PGThread(pg.pg, pg.tasks[ss]) as pts, PGThread(pg.pg, pg.tasks[es]) as pte:
    for i in jnp.arange(start_epoch - 1, epochs):
      if epochs > 1:
        pte.completed += 1
      for ii, x, p in zip(jnp.arange(steps), ds.xs[i], ds.pad[i]):
        pts.completed += 1
        yield DState(transform(x) if transform else x, p, *_form(i, ii), name=prefix)

    pte.completed, pts.completed = _form(i, ii)


def __test() -> None:
  """Test."""
  pg = PG.init_progress(extra_columns=(RESC,))
  keys = Keys.from_int_or_key(42)
  with pg:
    pg.console.print("Drop Last: False, Auto Pad: True")
    dl = IdxDataloader(314430, 256, drop_last=False)

    for i in es_loop(dl, pg, epochs=300, keys=keys, prefix="DFAT", start_epoch=10):
      # pg.update_res(
      #   "DFAT-S", {"epoch": i.epoch.item(), "step": i.step.item(), "name": i.name}
      # )
      continue
    pg.console.print(i)
    pg.console.print("Drop Last: True, Auto Pad: True")
    dl = IdxDataloader(10, 3, drop_last=True)
    for i in es_loop(dl, pg, keys, prefix="DTAT"):
      pg.console.print(i)


if __name__ == "__main__":
  __test()
