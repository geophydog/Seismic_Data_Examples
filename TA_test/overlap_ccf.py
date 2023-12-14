import numpy as np
from obspy import read
from numpy import ones, convolve
import re
import glob
import os
from obspy.core import Trace, AttribDict
import sys
from scipy.signal import hilbert
from scipy.signal.windows import hann

def process_trace(data, nfft, operator, p=0.01):
    d = data - np.mean(data); n = len(d)
    sn = int(n*p); hn = 2 * sn; h = hann(hn)
    d[:sn] *= h[:sn]; d[-sn:] *= h[-sn:]
    fd = np.fft.fft(d, nfft)
    fd /= np.convolve(np.abs(fd), operator, 'same')
    fd[np.isnan(np.abs(fd))] = 0
    return fd

argc = len(sys.argv)
if argc != 5:
    print('Usage: python %s yyyy-mm-dd channel1 channel2 sta.lst'%sys.argv[0])
    exit(1)
sta = {}
with open(sys.argv[4], 'r') as fin:
    for line in fin.readlines():
        tmp = line.strip().split('|')
        sta[tmp[0]+'.'+tmp[1]] = [float(tmp[3]), float(tmp[2])]

f1 = 0.01; f2 = 0.6
dt = 0.25; fs = 1 / dt
h = 20; N = 2 * h + 1
operator = np.ones(N) / N
segt = 60 * 60
ot = 30 * 60; pt = segt - ot
segn = int(segt/dt)
pn = int((segt-ot)/dt)
nfft = 2 * int(0.55*segt/dt) + 1
t = np.arange(nfft) * dt - nfft//2*dt
cor = Trace(t)
cor.stats.delta = dt
cor.stats.sac = AttribDict({})
cor.stats.sac.b = t[0]

ch1 = sys.argv[2]
ch2 = sys.argv[3]
dirs = sorted(glob.glob(sys.argv[1]+'*'))
for dd in dirs:
    print(dd)
    if dd[-1] != '/':
        dd += '/'
    try:
        st1 = read('%s/*H%s.SAC'%(dd, ch1))
        if ch2 == ch1:
            st2 = st1.copy()
        else:
            st2 = read('%s/*H%s.SAC'%(dd, ch2))
        sta1 = [tr.stats.station for tr in st1]
        sta2 = [tr.stats.station for tr in st2]
        st = []
        for s in sta1:
            if s in sta2:
                st.append([st1[sta1.index(s)], st2[sta2.index(s)]]) 
        lm = len(st)
        if lm < 2:
            continue
        utc1 = st[0][0].stats.starttime
        utc2 = st[0][0].stats.endtime
        date = '%s'%(str(utc1.date))
        for i in range(len(st)):
            tt11 = st[i][0].stats.starttime
            tt21 = st[i][1].stats.starttime
            tt12 = st[i][0].stats.endtime
            tt22 = st[i][1].stats.endtime
            if utc1 > tt11:
                utc1 = tt11
            if utc1 > tt21:
                utc1 = tt21
            if utc2 < tt12:
                utc2 = tt12
            if utc2 < tt22:
                utc2 = tt22
    except Exception:
        continue
    ccfdir = '%s%s_CCF_%s'%(ch1, ch2, date)
    if not os.path.exists(ccfdir):
        os.mkdir(ccfdir)
    nn = int((utc2-utc1)/dt) + 100
    data1 = np.zeros((lm, nn))
    data2 = np.zeros((lm, nn))
    ns = []; xy = []
    for i in range(lm):
        nb1 = int((st[i][0].stats.starttime-utc1)/dt)
        data1[i, nb1:nb1+len(st[i][0].data)] = st[i][0].data
        nb2 = int((st[i][1].stats.starttime-utc1)/dt)
        data2[i, nb2:nb2+len(st[i][1].data)] = st[i][1].data
        k = st[i][0].stats.network+'.'+st[i][0].stats.station
        ns.append(k)
        xy.append(sta[k])
    data_fd1 = []; data_fd2 = []
    for i in range(lm):
        dn1 = 0
        tmp1 = []; tmp2 = []
        while dn1 <= nn-segn:
            dn2 = dn1 + segn
            fd1 = process_trace(data1[i, dn1: dn2], nfft, operator)
            fd2 = process_trace(data2[i, dn1: dn2], nfft, operator)
            tmp1.append(fd1)
            tmp2.append(fd2)
            dn1 += pn
        data_fd1.append(tmp1); data_fd2.append(tmp2)
    nstack = len(data_fd1[0])
    for si in range(lm-1):
        for sj in range(si+1, lm):
            ccfname = 'COR_' + ns[si] +'_' + ns[sj] + '.SAC'
            ccfd = np.zeros(nfft, dtype=complex)
            for sk in range(nstack):
                ccfd += (data_fd1[si][sk]*np.conjugate(data_fd2[sj][sk]))
            cor.data = np.fft.ifftshift(np.fft.ifft(ccfd)).real
            cor.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
            cor.stats.sac.evlo = xy[si][0]
            cor.stats.sac.evla = xy[si][1]
            cor.stats.sac.stlo = xy[sj][0]
            cor.stats.sac.stla = xy[sj][1]
            cor.write(ccfdir+'/'+ccfname)
