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

from .data import DState, IdxDataloader, es_loop
from .keys import Keys, new_key
from .loops import PG, RESC, PGThread
from .metrics import (
  Accuracy,
  Metric,
  average_precision_score,
  convert,
  dice_coef,
  iou_coef,
  pr_auc_score,
  roc_auc_score,
)
from .preprocessing import (
  SCALER,
  identity_scaler,
  min_max_scaler,
  normalizer,
  select_scaler,
  standard_scaler,
)
from .states import BaseTrainState, state_fn
from .utils import (
  all_array,
  classit,
  filter_concat,
  pformat,
  rle,
  rle_array,
)

__version__ = "1.6.4"

__all__ = [
  "PG",
  "RESC",
  "SCALER",
  "Accuracy",
  "BaseTrainState",
  "DState",
  "IdxDataloader",
  "Keys",
  "Metric",
  "Metric",
  "PGThread",
  "all_array",
  "average_precision_score",
  "classit",
  "convert",
  "dice_coef",
  "es_loop",
  "filter_concat",
  "identity_scaler",
  "iou_coef",
  "min_max_scaler",
  "new_key",
  "normalizer",
  "pformat",
  "pr_auc_score",
  "rle",
  "rle_array",
  "roc_auc_score",
  "select_scaler",
  "standard_scaler",
  "state_fn",
]
