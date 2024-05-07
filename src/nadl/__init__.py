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
date     : Nov 29, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : __init__.py
project  : nadl
license  : GPL-3.0+

NADL
"""

from .data import DState, IdxDataloader
from .keys import Keys, new_key
from .loops import PG
from .metrics import dice_coef, iou_coef
from .preprocessing import (
  SCALER,
  identity_scaler,
  min_max_scaler,
  normalizer,
  select_scaler,
  standard_scaler,
)
from .utils import (
  classit,
  rle,
  rle_array,
)

__version__ = "1.0.1"

__all__ = [
  "PG",
  "SCALER",
  "DState",
  "IdxDataloader",
  "Keys",
  "classit",
  "dice_coef",
  "identity_scaler",
  "iou_coef",
  "min_max_scaler",
  "new_key",
  "normalizer",
  "rle",
  "rle_array",
  "select_scaler",
  "standard_scaler",
]
