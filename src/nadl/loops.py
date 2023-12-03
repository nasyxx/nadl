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


class PG(NamedTuple):
  """Progress."""

  pg: Progress
  console: Console
  tasks: dict[str, TaskID]


def init_progress(
  pg: Progress | None,
  console: Console | None,
  total: bool = True,
  extra_columns: tuple[ProgressColumn, ...] = (),
  show_progress: bool = True,
) -> PG:
  """Init progress bar."""
  if console is None:
    console = Console()
  if pg is None:
    pg = Progress(
      TextColumn(
        "{task.description}" + " - {task.completed}/{task.total}" if total else ""
      ),
      TimeRemainingColumn(),
      TimeElapsedColumn(),
      BarColumn(None),
      console=console,
      disable=not show_progress,
    )
  pg.columns = pg.columns + extra_columns
  return PG(pg, pg.console, {})
