import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from scipy.optimize import curve_fit
import gsd
import gsd.hoomd
matplotlib.use("TkAgg")

# print(matplotlib.rcParams.keys())

plt.rcParams['lines.markersize']        = 2
plt.rcParams['lines.linewidth']         = 1

plt.rcParams['mathtext.fontset']        = 'cm'
# plt.rcParams['font.family']             = 'sans-serif'
plt.rcParams['font.family']             = 'serif'
plt.rcParams['font.size']               = 10
plt.rcParams['font.serif']              = 'Times New Roman', 'Times', 'DejaVu Serif', 'serif'
plt.rcParams['font.sans-serif']         = 'Arial', 'DejaVu Sans', 'sans-serif'
plt.rcParams['text.usetex']             = False

plt.rcParams['axes.linewidth']          = 1
plt.rcParams['axes.labelsize']          = 10
plt.rcParams['axes.labelpad']           = 4
plt.rcParams['axes.formatter.limits']   = -3,4

plt.rcParams['xtick.top']               = True
plt.rcParams['xtick.bottom']            = True
plt.rcParams['xtick.labelsize']         = 9
plt.rcParams['xtick.minor.visible']     = True
plt.rcParams['xtick.direction']         = 'in'

plt.rcParams['ytick.right']             = True
plt.rcParams['ytick.left']              = True
plt.rcParams['ytick.labelsize']         = 9
plt.rcParams['ytick.minor.visible']     = True
plt.rcParams['ytick.direction']         = 'in'

plt.rcParams['legend.frameon']          = False
plt.rcParams['legend.fontsize']         = 7
plt.rcParams['legend.handletextpad']    = 0.2
plt.rcParams['legend.handlelength']     = 1.8
plt.rcParams['legend.columnspacing']    = 1
plt.rcParams['legend.labelspacing']     = 0.3
plt.rcParams['legend.borderpad']         = 0.2

plt.rcParams['figure.figsize']          = [3.37, 1.97]
plt.rcParams['figure.dpi']              = 600

plt.rcParams['savefig.pad_inches']      = 0.05
plt.rcParams['savefig.bbox']            = 'tight'

strFormat = '{:.2g}'

mar=['o','^','s','d','x','v','*','>','h']
col=['#440154','#31688e','#35b779','#e3cf21','#7ad151','#e3cf21','#e5c100','#858585','#000000']


plt.ion();


fig, ax = plt.subplots(num=100)
fig1, ax1 = plt.subplots(num=101)


data1 = np.genfromtxt('mean_Noli_F_vx_surfDensTop_fromPolar', delimiter=',')

Noli    = data1[:,0]
F   = data1[:,1]
vx   = data1[:,2]
surfDens = data1[:,3]
surfDens_polar = data1[:,4]
sigSurfDens = data1[:,5]
sigSurfDens_polar = data1[:,6]

Noli = Noli*5/(50*100*100 + Noli*5)

print(F.shape)


a = 0; b = a + 8; i=2
ax.errorbar(vx[a:b],surfDens[a:b], yerr=sigSurfDens[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],surfDens[a:b], yerr=sigSurfDens[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax1.set_ylim([60,80])

ax.set_xlabel(r"$v_x~[r_c/\tau]$")
ax.set_ylabel(r"$\Sigma~[r_c^{-2}]$")

ax.set_xscale('log')

ax.legend()

fig.savefig("figures/surfDensTopvsV.pdf")

a = 0; b = a + 8; i=2

sigY = np.sqrt( (sigSurfDens[a:b]/surfDens[a])**2 + (surfDens[a:b]/(surfDens[a]**2)*sigSurfDens[a])**2 )

ax1.errorbar(vx[a:b],surfDens[a:b]/surfDens[a], yerr=sigY, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )


a = b; b = a + 8; i=i+1

sigY = np.sqrt( (sigSurfDens[a:b]/surfDens[a])**2 + (surfDens[a:b]/(surfDens[a]**2)*sigSurfDens[a])**2 )

ax1.errorbar(vx[a:b],surfDens[a:b]/surfDens[a], yerr=sigY, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

ax1.text(0.9e-3, 1.03, '(b)', fontsize=12)

# ax1.set_ylim([60,80])

ax1.set_xlabel(r"$v_x~[r_c/\tau]$")
ax1.set_ylabel(r"$\Sigma/\Sigma_0$")

ax1.set_xscale('log')

ax1.legend()

fig1.savefig("figures/surfDensTopvsV_Norm.pdf")


input("press enter")
