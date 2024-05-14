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
date     : Nov 30, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : loops.py
project  : nadl
license  : GPL-3.0+

Train Eval Loops
"""

from collections.abc import Hashable, Mapping

import jax
from equinox import Module

from types import TracebackType
from typing import Any, Self

from rich.console import Console
from rich.progress import (
  BarColumn,
  Progress,
  ProgressColumn,
  TaskID,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)
from rich.theme import Theme

DEF_LIGHT_THEME = Theme({
  "bar.back": "#50616D",
  "bar.complete": "#D3CBAF",
  "bar.finished": "#b49b7f",
})


class PG(Module):
  """Progress."""

  pg: Progress
  console: Console
  tasks: dict[Hashable, TaskID]

  @classmethod
  def init_progress(
    cls: type[Self],
    pg: Progress | None = None,
    console: Console | None = None,
    total: bool = True,
    bar_width: int | None = 20,
    extra_columns: tuple[ProgressColumn, ...] = (),
    show_progress: bool = True,
    theme: Theme | None = None,
  ) -> Self:
    """Init progress bar."""
    if console is None:
      console = Console(theme=theme or DEF_LIGHT_THEME)
    if pg is None:
      pg = Progress(
        TextColumn(
          "{task.description}" + " - {task.completed}/{task.total}" if total else ""
        ),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        BarColumn(bar_width),
        TextColumn("{task.fields[res]}"),
        console=console,
        disable=not show_progress,
      )
    pg.columns += extra_columns
    return cls(pg, console, {})

  def add_columns(self, columns: tuple[ProgressColumn, ...]) -> Self:
    """Add columns."""
    self.pg.columns += columns
    return self

  def add_task(
    self,
    description: str,
    start: bool = True,
    total: float | None = 100,
    completed: int = 0,
    visible: bool = True,
    **fileds: Any,  # noqa: ANN401
  ) -> TaskID:
    """Add task."""
    task = self.pg.add_task(
      description,
      start=start,
      total=total,
      completed=completed,
      visible=visible,
      **fileds,
    )
    self.tasks[description] = task
    return task

  def advance(self, task: TaskID, advance: float = 1) -> None:
    """Advance task."""
    self.pg.advance(task, advance=advance)

  def __enter__(self) -> Progress:
    """Enter."""
    return self.pg.__enter__()

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Exit."""
    self.pg.__exit__(exc_type, exc_val, exc_tb)

  @staticmethod
  def pformat(xs: Mapping[str, float | int | str | None]) -> str:
    """Pretty format progress results."""
    with (console := Console()).capture() as capture:
      nxs = jax.tree.map(lambda x: float(f"{x:.4f}") if isinstance(x, float) else x, xs)
      console.print(nxs, soft_wrap=True, justify="left", no_wrap=True, width=40)
    return capture.get()

  def update_res(
    self, name: str, updates: Mapping[str, float | int | str | None]
  ) -> None:
    """Update res."""
    if name in self.tasks:
      self.pg.update(self.tasks[name], res=self.pformat(updates))


def test() -> None:
  """Test progress."""
  import time  # noqa: PLC0415

  pg = PG.init_progress()
  with pg:
    t0 = pg.add_task("Task 0", total=100)
    for _i in range(100):
      pg.advance(t0)
      time.sleep(0.5)


if __name__ == "__main__":
  test()
