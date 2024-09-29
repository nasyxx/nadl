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
date     : May 14, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : states.py
project  : nadl
license  : GPL-3.0+

States
"""

from __future__ import annotations

import shutil
from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import optax
import orbax.checkpoint as ocp

from jaxtyping import PyTree
from typing import TYPE_CHECKING, cast, Self

if TYPE_CHECKING:
  from pathlib import Path

  import jax

  from rich.console import Console


_sentinel = cast(eqx.nn.State, object())


class BaseTrainState[T, M](eqx.Module):
  """Train state."""

  model: M
  tx: optax.GradientTransformation
  opt_state: optax.OptState
  loss: jax.Array
  step: jax.Array
  conf: T

  @classmethod
  @abstractmethod
  def init[**P](cls: type[Self], *args: P.args, **kwds: P.kwargs) -> Self:
    """Create initial state."""
    raise NotImplementedError

  def apply_grads(self, loss: jax.Array, grads: eqx.Module) -> BaseTrainState[T, M]:
    """Apply gradients."""
    updates, opt_state = self.tx.update(
      cast(optax.Updates, grads), self.opt_state, params=cast(optax.Params, self.model)
    )
    model = eqx.apply_updates(self.model, updates)
    return eqx.tree_at(
      lambda x: (x.model, x.opt_state, x.loss, x.step),
      self,
      (model, opt_state, loss, self.step + 1),
    )


class BaseTrainStateWS[T, M](eqx.Module):
  """Train state."""

  model: M
  state: eqx.nn.State
  tx: optax.GradientTransformation
  opt_state: optax.OptState
  loss: jax.Array
  step: jax.Array
  conf: T

  @classmethod
  @abstractmethod
  def init[**P](cls: type[Self], *args: P.args, **kwds: P.kwargs) -> Self:
    """Create initial state."""
    raise NotImplementedError

  def apply_grads(
    self, loss: jax.Array, grads: eqx.Module, state: eqx.nn.State
  ) -> BaseTrainStateWS[T, M]:
    """Apply gradients."""
    updates, opt_state = self.tx.update(
      cast(optax.Updates, grads), self.opt_state, params=cast(optax.Params, self.model)
    )
    model = eqx.apply_updates(self.model, updates)
    return eqx.tree_at(
      lambda x: (x.model, x.state, x.opt_state, x.loss, x.step),
      self,
      (model, state, opt_state, loss, self.step + 1),
    )


type T_savefn = Callable[[int, BaseTrainState, PyTree, PyTree], None]


def state_fn(
  rpath: Path,
  console: Console | None = None,
  keeps: int = 5,
  clean: bool = False,
  item_names: tuple[str, ...] | None = None,
  item_handlers: dict | None = None,
  best_fn: Callable[[PyTree], float] | None = None,
) -> tuple[ocp.CheckpointManager, T_savefn]:
  """Get states manager."""
  match (item_names, item_handlers):
    case (None, None):
      item_names = ("state", "extra_metadata")
      item_handlers = {
        "state": ocp.PyTreeCheckpointHandler(),
        "extra_metadata": ocp.PyTreeCheckpointHandler(),
      }
    case _ if item_names and item_handlers:
      for i in item_names:
        assert i in item_handlers, f"Item {i} not in item_handlers."
    case _:
      raise ValueError("item_names and item_handlers should be both None or not None.")

  if console:
    console.log(f"Checkpoint path at {rpath}")
  if rpath.exists() and clean:
    if console:
      console.log("Cleaning up checkpoint...")
    shutil.rmtree(rpath)

  mngr = ocp.CheckpointManager(
    rpath,
    options=ocp.CheckpointManagerOptions(
      max_to_keep=keeps, save_interval_steps=1, best_fn=best_fn
    ),
    item_names=item_names,
    item_handlers=item_handlers,
  )

  def save(
    step: int, state: BaseTrainState, metadata: PyTree = None, metrics: PyTree = None
  ) -> None:
    """Save state."""
    mngr.save(
      step,
      args=ocp.args.Composite(
        state=ocp.args.PyTreeSave(state),  # type: ignore
        extra_metadata=ocp.args.PyTreeSave(metadata or {}),  # type: ignore
      ),
      metrics=metrics or {},
    )

  return mngr, save
