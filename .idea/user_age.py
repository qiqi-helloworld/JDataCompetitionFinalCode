#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "QIQI"

import pandas as pd
import numpy as np
user = pd.read_csv("./data/JData_User.csv")
print (np.unique(np.array(user['age'])))