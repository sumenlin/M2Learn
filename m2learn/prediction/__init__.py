#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# M2Learn: An python library for multi-modal data learning.
# - Homepage: https://github.com/sumenlin/M2Learn/
# - Author: Suwen Lin and contributors
# - License: New BSD
# - Date: February 2019
# """
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Suwen"
__version__ = "0.1.0"

from .regressor import *
from .classifier import *
from .ensemble import *


__all__ = ('ensembleClassifier',
           'oneRegressor',
           'ensembleRegressor',
           'oneClassifier'
           )

