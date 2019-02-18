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
    from .preprocessing import *
    # from m2learn import preprocessing
except ImportError:
    pass

try:
    from .feature import *
    # from m2learn import feature
except ImportError:
    pass

try:
    from .prediction import *
    # from m2learn import prediction
except ImportError:
    pass

try:
    # from .pipeline import *
    from m2learn import pipeline
except ImportError:
    pass

