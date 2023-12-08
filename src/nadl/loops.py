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
from typing import NamedTuple
from collections.abc import Hashable

from rich.console import Console
from rich.progress import (
  Progress,
  TaskID,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  BarColumn,
  ProgressColumn,
)
from rich.theme import Theme


DEF_LIGHT_THEME = Theme(
  {
    "bar.back": "#50616D",
    "bar.complete": "#789262",
    "bar.finished": "#057748",
  }
)


class PG(NamedTuple):
  """Progress."""

  pg: Progress
  console: Console
  tasks: dict[Hashable, TaskID]


def init_progress(
  pg: Progress | None = None,
  console: Console | None = None,
  total: bool = True,
  bar_width: int | None = 20,
  extra_columns: tuple[ProgressColumn, ...] = (),
  show_progress: bool = True,
  theme: Theme | None = None,
) -> PG:
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
  pg.columns = pg.columns + extra_columns
  return PG(pg, pg.console, {})


def add_columns(pg: PG, columns: tuple[ProgressColumn, ...]) -> PG:
  """Add columns."""
  pg.pg.columns = pg.pg.columns + columns
  return pg


def test() -> None:
  """Test progress."""
  import time
  pg = init_progress()
  with pg.pg:
    t0 = pg.tasks[0] = pg.pg.add_task("Task 0", total=100)
    for i in range(100):
      pg.pg.advance(t0)
      time.sleep(0.5)



if __name__ == "__main__":
  test()
