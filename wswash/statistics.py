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

    # find indexes for watlev < 0
    for i in range(len(neg)-1):
        if neg[i+1] != neg[i] + 1:
            neg1.append(neg[i])

    # fix: last wave if watlev > 0
    if watlev[-1]>0:
        neg1.append(neg[-1])

    # find upcrossing indexes with least squares
    for i in range(len(neg1)):
        p = np.polyfit(time[slice(neg1[i],neg1[i]+1)], watlev[slice(neg1[i], neg1[i]+1)], 1)
        r.append(np.roots(p)[0])
    r = np.abs(np.array(r))

    # calculate H for each wave
    for i in np.arange(1, len(r), 1):
        H.append(max(watlev[np.where((time < r[i]) & (time > r[i-1]))]) - min(watlev[np.where((time < r[i]) & (time > r[i-1]))]))

    # calculate T for each wave
    T = np.diff(r)

    return(T, H)

