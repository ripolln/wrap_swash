#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def upcrossing(time, watlev):
    '''
    Performs fast fourier transformation FFt by extracting
    the wave series mean and interpolating to find the cross-zero

    time   - time series
    watlev - surface elevation series

    return - Wave heigths and periods
    '''

    watlev = watlev - np.mean(watlev)
    neg = np.where(watlev < 0)[0]

    neg1 = []
    r = []
    H = []

    for i in range(len(neg)-1):
        if neg[i+1] != neg[i] + 1:
            neg1.append(neg[i])

    for i in range(len(neg1)-1):
        p = np.polyfit(time[slice(neg1[i],neg1[i]+1)], watlev[slice(neg1[i], neg1[i]+1)], 1)
        r.append(np.roots(p))

    r = [item for sublist in r for item in sublist]
    r = np.abs(r)
    for i in np.arange(2, len(r), 1):
        H.append(max(watlev[np.where((time < r[i]) & (time > r[i-1]))]) - min(watlev[np.where((time < r[i]) & (time > r[i-1]))]))

    T = np.diff(r)

    return(T, H)

