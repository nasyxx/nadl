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

from __future__ import annotations

from threading import Event, Thread

from equinox import Module

from typing import TYPE_CHECKING, Any, Self

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

from .utils import pformat

if TYPE_CHECKING:
  from collections.abc import Hashable, Mapping

  from types import TracebackType

DEF_LIGHT_THEME = Theme({
  "bar.back": "#50616D",
  "bar.complete": "#D3CBAF",
  "bar.finished": "#b49b7f",
})


RESC = TextColumn("{task.fields[res]}")


class PGThread(Thread):
  """Progress thread."""

  def __init__(self, pg: Progress, task_id: TaskID) -> None:
    """Init."""
    super().__init__()
    self.pg = pg
    self.tid = task_id
    self.done = Event()
    self.completed = 0
    self.res: str | None = None
    super().__init__()

  def run(self) -> None:
    """Run."""
    last_completed = 0
    wait = self.done.wait
    while not wait(0.2):
      if (completed := self.completed) != last_completed:
        self.pg.advance(self.tid, completed - last_completed)
        last_completed = completed
        if self.res is not None:
          self.pg.update(self.tid, res=self.res)
          self.res = None
    self.pg.update(self.tid, completed=self.completed, refresh=True)

  def __enter__(self) -> Self:
    """Enter."""
    self.start()
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Exit."""
    del exc_type, exc_val, exc_tb
    self.done.set()
    self.join()


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

  def update_res(
    self, name: str, updates: Mapping[str, float | int | str | None]
  ) -> None:
    """Update res."""
    if name in self.tasks:
      self.pg.update(self.tasks[name], res=pformat(updates))


def test() -> None:
  """Test progress."""
  pg = PG.init_progress()
  t0 = pg.add_task("Task 0", total=10**8)
  with pg, PGThread(pg.pg, t0) as pt:
    for _i in range(10**8):
      pt.completed += 1


if __name__ == "__main__":
  test()
