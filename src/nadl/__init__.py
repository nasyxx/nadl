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

from .keys import (
  Keys,
  from_int_or_key,
  from_state,
  new_key,
  next_key,
  reverse,
  take,
)
from .loops import PG, init_progress
from .losses import dice_loss, softmax_focal_loss, sigmoid_focal_loss
from .metrics import (
  accuracy,
  auc,
  average_precision_score,
  confusion_matrix,
  dice_coef,
  f1_score,
  iou,
  precision_recall_curve,
  precision_recall_fscore_support,
  precision_score,
  recall_scroe,
  roc_auc_score,
  roc_curve,
  support,
)
from .preprocessing import (
  SCALER,
  identity_scaler,
  min_max_scaler,
  standard_scaler,
  normalizer,
  select_scaler,
)
from .utils import (
  classit,
  rle,
  rle_array,
)
# from .blocks import (
#   BasicBlock,
#   BottleneckBlock,
#   ConvBlock,
#   DownSample,
#   ResNetBasic,
#   ResNetBottleneck,
#   UpSample,
#   make_layer,
# )

__version__ = "0.5.1"

__all__ = [
  # "BasicBlock",
  # "BottleneckBlock",
  # "ConvBlock",
  # "DownSample",
  "Keys",
  "PG",
  # "ResNetBasic",
  # "ResNetBottleneck",
  "SCALER",
  # "UpSample",
  "accuracy",
  "auc",
  "average_precision_score",
  "classit",
  "confusion_matrix",
  "dice_coef",
  "dice_loss",
  "f1_score",
  "from_int_or_key",
  "from_state",
  "identity_scaler",
  "init_progress",
  "iou",
  # "make_layer",
  "min_max_scaler",
  "new_key",
  "next_key",
  "normalizer",
  "precision_recall_curve",
  "precision_recall_fscore_support",
  "precision_score",
  "recall_scroe",
  "reverse",
  "rle",
  "rle_array",
  "roc_auc_score",
  "roc_curve",
  "select_scaler",
  "sigmoid_focal_loss",
  "softmax_focal_loss",
  "standard_scaler",
  "support",
  "take",
]
