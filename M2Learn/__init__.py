#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M2Learn: An python library for multi-modal data learning.
- Homepage: https://github.com/sumenlin/M2Learn/
- Author: Suwen Lin and contributors
- License: New BSD
- Date: February 2019
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Suwen"
__version__ = "0.1.0"

try:
    # from .Arms import *
    from M2Learn import preprocessing
except ImportError:
    pass

try:
    from M2Learn import feature
except ImportError:
    pass

try:
    # from .Policies import *
    from M2Learn import prediction
except ImportError:
    pass

try:
    # from .Policies import *
    from M2Learn import pipeline
except ImportError:
    pass