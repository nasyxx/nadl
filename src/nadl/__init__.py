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

from .blocks import FastKANLayer, RadialBasisFunction, SplineLinear
from .data import DState, IdxDataloader, es_loop
from .keys import Keys, new_key
from .loops import PG, RESC, PGThread
from .metrics import (
  Accuracy,
  Metric,
  average_precision_score,
  convert,
  dice_coef,
  info_nce,
  iou_coef,
  pr_auc_score,
  roc_auc_score,
)
from .nets import FastKAN, pMTnet
from .preprocessing import (
  SCALER,
  identity_scaler,
  min_max_scaler,
  normalizer,
  select_scaler,
  standard_scaler,
)
from .states import BaseTrainState, T_savefn, state_fn
from .utils import (
  all_array,
  batch_array_p,
  classit,
  filter_concat,
  filter_tree,
  pformat,
  rle,
  rle_array,
)

__version__ = "1.9.0"

__all__ = [
  "PG",
  "RESC",
  "SCALER",
  "Accuracy",
  "BaseTrainState",
  "DState",
  "FastKAN",
  "FastKANLayer",
  "IdxDataloader",
  "Keys",
  "Metric",
  "Metric",
  "PGThread",
  "RadialBasisFunction",
  "SplineLinear",
  "T_savefn",
  "all_array",
  "average_precision_score",
  "batch_array_p",
  "classit",
  "convert",
  "dice_coef",
  "es_loop",
  "filter_concat",
  "filter_tree",
  "identity_scaler",
  "info_nce",
  "iou_coef",
  "min_max_scaler",
  "new_key",
  "normalizer",
  "pMTnet",
  "pformat",
  "pr_auc_score",
  "rle",
  "rle_array",
  "roc_auc_score",
  "select_scaler",
  "standard_scaler",
  "state_fn",
]
