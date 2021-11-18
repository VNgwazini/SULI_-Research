#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:05:03 2019

@author: 1vn
"""

import pywt
import matplotlib.pyplot as plt
import numpy as np

#grab first 16 values
ts = [
      1083.758226,
1094.575603,
1089.414772,
1095.754668,
1085.904208,
1095.008753,
1090.325359,
1084.254191,
1084.521566,
1088.267813,
1082.149365,
1132.168228,
1097.570638,
1087.306891,
1081.729159,
1083.464893]

(ca, cd) = pywt.dwt(ts,'haar')

cat = pywt.threshold(ca, np.std(ca)/2,mode='soft')
cdt = pywt.threshold(cd, np.std(cd)/2,mode='soft')

#reconstruct signal and store in varible ts_rec for access later
ts_rec = pywt.idwt(cat, cdt, 'haar')

plt.close('all')

plt.subplot(211)


# Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')

# Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid('on')

plt.subplot(212)
plt.plot(ts)
plt.plot(ts_rec, 'r')
plt.legend(['original signal', 'reconstructed signal'], loc=7)
plt.grid('on')
plt.show()