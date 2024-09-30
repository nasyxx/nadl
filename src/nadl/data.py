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

Simple dataset and dataloader version 2.
"""

from collections.abc import Callable, Iterator
from warnings import warn

import jax
import jax.numpy as jnp
from equinox import Module, field, filter_jit, filter_vmap
from equinox.nn import Identity

from jaxtyping import Array, Bool, Int, PRNGKeyArray
from typing import NamedTuple, Protocol, cast

import numpy as np

from einops import repeat

from .keys import Keys
from .loops import PG, RESC, PGThread

type _IDX_FN = Callable[[Int[Array, ""]], DState[Int[Array, " b d"]]]


class DState[T](NamedTuple):
  """Dataloader state."""

  xs: T
  pad: Bool[Array, " b"]
  length: Int[Array, ""]
  epoch: Int[Array, ""] = jnp.asarray(0)
  step: Int[Array, ""] = jnp.asarray(0)
  key: PRNGKeyArray = field(default_factory=lambda: jax.random.key(42))
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


def batch_index(
  length: int,
  batch_size: int,
  drop_last: bool = False,
  shuffle: bool = False,
  *,
  key: PRNGKeyArray | None = None,
) -> _IDX_FN:
  """Batchify the index."""
  new_length = length if not drop_last else length - length % batch_size
  pad = (batch_size - r) % batch_size if (r := new_length % batch_size) else 0
  pad = pad if pad != batch_size else 0
  drop_num = length % batch_size if drop_last else 0
  _idxes = jnp.arange(length)
  dlength = jnp.asarray(
    length // batch_size + (1 if (not drop_num) and length % batch_size else 0)
  )
  depoch = jnp.asarray(0)
  if key is None:
    warn("Key is not provided, using 42 as random key seed.", stacklevel=1)
    key = jax.random.key(42)

  @filter_jit
  def _index(epoch: Int[Array, ""] = depoch) -> DState[Int[Array, " b d"]]:
    assert epoch.ndim == 0, "Epoch must be a scalar."
    new_key = jax.random.fold_in(key, epoch)
    if shuffle:
      idxes = jnp.take_along_axis(
        _idxes,
        # NOTE: Fallback to numpy argsort since it has performance isssue in CPU.
        # https://github.com/google/jax/issues/10434
        fallback_argsort(jax.random.uniform(new_key, (length,))),
        axis=0,
      )
    else:
      idxes = _idxes

    _len = length if not drop_last else length - drop_num
    idxes = jnp.r_[idxes, jnp.full(pad, -1, idxes.dtype)]
    idxes = idxes[: _len + pad].reshape(-1, batch_size)
    return DState(
      idxes,
      jnp.where(idxes == -1, 1, 0).astype(bool),
      dlength,
      epoch,
      key=jax.random.split(new_key, idxes.shape[0]),
    )

  return _index


class TransT[_I, _O](Protocol):
  """Transform protocol."""

  def __call__(self, x: _I, *, key: PRNGKeyArray | None) -> _O:
    """Transform Forwardl."""
    ...


class DataLoader[T](Module):
  """Simple data loader."""

  gen: _IDX_FN
  embed: TransT[Int[Array, " d"], T]
  transform: TransT[T, T]
  _default_epoch: Int[Array, ""]
  _length: int
  _data_length: int
  _batch_size: int

  def __init__(
    self,
    length: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    key: PRNGKeyArray | None = None,
    *,
    embed: TransT[Int[Array, " d"], T] | None = None,
    transform: TransT[T, T] | None = None,
  ) -> None:
    """Initiate the dataloader."""
    self.gen = batch_index(length, batch_size, drop_last, shuffle, key=key)
    self.embed = (
      embed if embed is not None else cast(TransT[Int[Array, " d"], T], Identity())
    )
    self.transform = (
      transform if transform is not None else cast(TransT[T, T], Identity())
    )
    # why jit make it slower?
    # self.embed = filter_jit(self.embed)
    # self.transform = filter_jit(self.transform)
    self._default_epoch = jnp.asarray(0)
    self._length = self.gen(self._default_epoch).length.item()
    self._data_length = length
    self._batch_size = batch_size

  def __call__(self, epoch: Int[Array, ""] | None = None) -> Iterator[DState[T]]:
    """Get the indexes."""
    data = (
      filter_jit(self.gen)(self._default_epoch) if epoch is None else self.gen(epoch)
    )
    for step, d, p, k in zip(jnp.arange(len(self)) + 1, data.xs, data.pad, data.key):
      yield DState(
        self.transform(self.embed(d, key=k), key=k),
        p,
        data.length,
        data.epoch,
        step,
        k,
      )

  @filter_jit
  def _ex(self, x: Array) -> Array:
    """Ex epoch and steps.

    epoch len *batch -> (epoch len) * batch
    """
    if x.ndim > 1:
      return x.reshape(-1, *x.shape[2:])
    return repeat(x, "e ... -> (e l) ...", l=self._length)

  def epoch_iter(
    self, epoch_start: int, epoch_end: int | None = None
  ) -> Iterator[DState[T]]:
    """Iterate with epochs."""
    gen = filter_jit(filter_vmap(self.gen))
    # steps = jnp.arange(self._length)
    if epoch_end is None:
      epoch_end = epoch_start + 1
      epoch_start = 1
    data = gen(jnp.arange(epoch_start, epoch_end))

    data = jax.tree.map(self._ex, data)
    for step, x, pad, length, epoch, key in zip(
      jnp.arange(data.xs.shape[0]) + 1,
      data.xs,
      data.pad,
      data.length,
      data.epoch,
      data.key,
    ):
      yield DState(
        self.transform(self.embed(x, key=key), key=key),
        pad,
        length,
        epoch,
        step,
        key,
      )

  def viter(
    self, epochs: int, chunks: int = 100, *, epoch_bias: int = 1
  ) -> Iterator[DState[T]]:
    """Iterate with epochs and chunks."""
    for chunk in range(epoch_bias, epochs + epoch_bias, chunks):
      yield from self.epoch_iter(chunk, min(chunk + chunks, epochs + epoch_bias))

  def __len__(self) -> int:
    """Get the length."""
    return self._length


def es_loop[T](
  dl: DataLoader[T],
  pg: PG,
  epochs: int = 1,
  start_epoch: int = 1,
  chunks: int = 100,
  prefix: str = "L",
  es: str = "E",
  ss: str = "S",
) -> Iterator[DState[T]]:
  """Simple epoch loop."""
  assert epochs > 0, "Epochs should be greater than 0."
  es, ss = f"{prefix}-{es}", f"{prefix}-{ss}"
  if es in pg.tasks:
    pg.pg.reset(pg.tasks[es], total=epochs)
  else:
    pg.add_task(es, total=epochs, res="", visible=epochs > 1)
  pg.advance(pg.tasks[es], start_epoch - 1)
  if ss in pg.tasks:
    pg.pg.reset(pg.tasks[ss], total=len(dl) * epochs, res="")
  else:
    pg.add_task(ss, total=len(dl) * epochs, res="")
  pg.advance(pg.tasks[ss], (start_epoch - 1) * len(dl))

  with PGThread(pg.pg, pg.tasks[ss]) as pts, PGThread(pg.pg, pg.tasks[es]) as pte:
    epoch = start_epoch
    for d in dl.viter(epochs, chunks, epoch_bias=start_epoch):
      if d.epoch != epoch:
        pte.completed += 1
        epoch = d.epoch
      pts.completed += 1
      yield d


def __test() -> None:
  """Test."""
  pg = PG.init_progress(extra_columns=(RESC,))
  keys = Keys.from_int_or_key(42)

  pg.console.print("Drop Last: False, Auto Pad: True")

  dl = DataLoader(314430, 256, drop_last=False, shuffle=True, key=keys(0))
  with pg:
    for d in es_loop(dl, pg, epochs=2):  # noqa: B007
      continue
    for d in es_loop(dl, pg, epochs=1):  # noqa: B007
      continue
    pg.console.print(jax.tree.map(jnp.shape, d))  # type: ignore
  # for epoch in tqdm(jnp.arange(300)):
  #   for i in dl(epoch):
  #     continue
  #   continue
  # for d in tqdm(dl.viter(300), total=300 * len(dl)):
  #   continue
  # pg.console.print(i)
  # pg.console.print("Drop Last: True, Auto Pad: True")
  # dl = IdxDataloader(10, 3, drop_last=True)
  # for i in es_loop(dl, pg, keys, prefix="DTAT"):
  #   pg.console.print(i)


if __name__ == "__main__":
  __test()
