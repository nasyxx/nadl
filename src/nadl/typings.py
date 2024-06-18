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
date     : Jun 17, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : typings.py
project  : nadl
license  : GPL-3.0+

Typings for NADL.
"""

from jaxtyping import Array, Float, Int, Key, Num, PRNGKeyArray

type F = Float[Array, "..."]
type FB = Float[Array, "B ..."]
type Ia = Int[Array, "..."]
type IaB = Int[Array, "B ..."]
type K = Key[PRNGKeyArray, "..."]
type KB = Key[PRNGKeyArray, "B ..."]
type N = Num[Array, "..."]
type NB = Num[Array, "B ..."]
