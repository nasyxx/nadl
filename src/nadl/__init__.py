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

from . import data_v1 as datav1
from .blocks import FastKANLayer, RadialBasisFunction, SplineLinear
from .data import DataLoader, DState, TransT, batch_index, es_loop, fallback_argsort
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
from .nets import (
  FastKAN,
  pMTnet,
  resnet18,
  resnet34,
  resnet50,
  resnet101,
  resnet152,
  resnext50_32x4d,
  resnext101_32x8d,
  resnext101_64x4d,
  wide_resnet50_2,
  wide_resnet101_2,
)
from .preprocessing import (
  SCALER,
  identity_scaler,
  min_max_scaler,
  normalizer,
  select_scaler,
  standard_scaler,
)
from .states import BaseTrainState, BaseTrainStateWS, T_savefn, state_fn
from .surgery import (
  get_bias,
  get_weight,
  init_fn,
  init_surgery,
  is_conv,
  is_conv1d,
  is_conv2d,
  is_linear,
  kaiming_init,
)
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

__version__ = "1.12.6"

__all__ = [
  "PG",
  "RESC",
  "SCALER",
  "Accuracy",
  "BaseTrainState",
  "DState",
  "DataLoader",
  "FastKAN",
  "FastKANLayer",
  "Keys",
  "Metric",
  "Metric",
  "PGThread",
  "RadialBasisFunction",
  "SplineLinear",
  "T_savefn",
  "TransT",
  "all_array",
  "average_precision_score",
  "batch_array_p",
  "batch_index",
  "classit",
  "convert",
  "datav1",
  "dice_coef",
  "es_loop",
  "fallback_argsort",
  "filter_concat",
  "filter_tree",
  "get_bias",
  "get_weight",
  "identity_scaler",
  "info_nce",
  "init_fn",
  "init_surgery",
  "iou_coef",
  "is_conv",
  "is_conv1d",
  "is_conv2d",
  "is_linear",
  "kaiming_init",
  "min_max_scaler",
  "new_key",
  "normalizer",
  "pMTnet",
  "pformat",
  "pr_auc_score",
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "resnet152",
  "resnext50_32x4d",
  "resnext101_32x8d",
  "resnext101_64x4d",
  "rle",
  "rle_array",
  "roc_auc_score",
  "select_scaler",
  "standard_scaler",
  "state_fn",
  "wide_resnet50_2",
  "wide_resnet101_2",
]
