import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, hann
from obspy import read
import sys

argc = len(sys.argv)
if argc != 8:
    print('Usage: python %s SAC T1(s) T2(s) gv1(km/s) gv2(km/s) ch(R/Z) CCF/GF'%sys.argv[0])
    exit(1)

tr = read(sys.argv[1])[0]
ch = sys.argv[6]
tag = sys.argv[7]
d0 = tr.data.copy()
if tag == 'CCF':
    n = len(tr.data)
    hn = n // 2
    pm1 = np.abs(tr.data[:hn]).max()
    pm2 = np.abs(tr.data[hn+1:]).max()
    if pm1 > pm2:
        tr.data = tr.data[::-1]
    tr.differentiate()
    tr.data *= (-1)
else:
    if ch == 'Z':
        tr.differentiate()
T1 = float(sys.argv[2])
T2 = float(sys.argv[3])
gv1 = float(sys.argv[4])
gv2 = float(sys.argv[5])
dt = tr.stats.delta
b = tr.stats.sac.b
if 'gcarc' in tr.stats.sac:
    gcarc = tr.stats.sac.gcarc * 111.195
elif 'dist' in tr.stats.sac:
    gcarc = tr.stats.sac.dist
else:
    print('No distance data!!!')
    trc = read('all_ncf/'+sys.argv[1].split('/')[-1])[0]
    gcarc = trc.stats.sac.dist
t1 = gcarc / gv2
t2 = gcarc / gv1
n1 = int((t1-b)/dt); n2 = int((t2-b)/dt)
nt = n2 - n1 + 1
tr.data[:n1] = 0.
tr.data[n2:] = 0.
d = tr.data[n1: n2+1]
t = np.arange(nt) * dt + t1

dT0 = 0.5
T = np.arange(T1, T2+dT0, dT0)
nT = len(T)
dT = 0.25
ng = 251
dv = 0.2
p = np.zeros((ng, nT))
vph = np.linspace(gv1, gv2, ng)
dn = int(dv/(vph[1]-vph[0]))
h = hann(2*dn)
for i in range(nT):
    dTT = dT + i * 5e-3
    f1 = 1 / (T[i]+dTT)
    f2 = 1 / (T[i]-dTT)
    trc = tr.copy()
    trc.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
    d = trc.data[n1: n2+1]
    d /= abs(d).max()
    tmp = gcarc/(t-T[i]/8)
    index = np.argsort(tmp)
    natural = CubicSpline(tmp[index], d[index], bc_type='natural')
    p[:, i] = natural(vph, nu=0)
    p[:dn, i] *= h[:dn]; p[-dn:, i] *= h[dn:]
    p[:, i] /= np.abs(p[:, i]).max()
    p[:dn, i] *= h[:dn]; p[-dn:, i] *= h[dn:]

d0 /= np.abs(d0).max()
d1 = tr.data; d1 /= np.abs(d1).max()

fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=1)
ax1.plot(tr.data[n1: n2+1], t, lw=1.25, color='b', alpha=0.5, label='EGF')
ax1.plot(d0[n1: n2+1], t, lw=1, color='r', alpha=0.5, label='CCF')
plt.legend(fontsize=14, loc='upper right')
ax1.set_ylabel('Lag time (s)')
ax1.set_title('Inter-sta dist: %.2f km'%gcarc)
ax1.set_ylim(t[0], t[-1])
ax1.invert_yaxis()

ax2 = plt.subplot2grid((1, 4), (0, 1), rowspan=1, colspan=3)
ax2.pcolormesh(T, vph, p, cmap='RdYlBu_r')
ax2.plot(T, gcarc/T, 'k--', lw=2, label='1 $\lambda$')
ax2.plot(T, gcarc/T*0.65, 'm--', lw=2, label='0.65 $\lambda$')
ax2.plot(T, gcarc/T/2, 'g--', lw=2, label='1/2 $\lambda$')
ax2.plot(T, gcarc/T/3, 'y--', lw=2, label='1/3 $\lambda$')
plt.legend()
ax2.set_xlabel('Period (s)')
ax2.set_ylabel('Phase velocity (km/s)')
ax2.set_xlim(T1, T2)
ax2.set_ylim(gv1, gv2)
ax2.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')

dv = 0.075; r = 1.0
tmp = plt.ginput()
if len(tmp) < 1:
    plt.close()
    sys.exit(1)
    plt.close()
t0, v0 = tmp[0]
print(t0, v0)
tn0 = int((t0-T.min())/(T[1]-T[0])+0.5)
vn0 = int((v0-vph.min())/(vph[1]-vph[0])+0.5)
vi = vn0
vm = []
for i in range(tn0, nT):
    dvn = int(dv/(vph[1]-vph[0])+0.5)
    pi = np.argmax(p[vi-dvn: vi+dvn, i])
    vi = vi - dvn + pi
    vm.append([T[i], vph[vi]])
    if gcarc/T[i]*r < vph[vi]:
        break
vm = np.array(vm)
ax2.plot(vm[:, 0], vm[:, 1], 'k.', lw=2, label='measured vph')

plt.show()
