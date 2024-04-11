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


data1 = np.genfromtxt('mean_Noli_F_vx_adv_rec_semiA_semiB_rotTh_subsH_ridgeHL_ridgeHR_brushHL_brushHR', delimiter=',')

Noli    = data1[:,0]
F   = data1[:,1]
vx   = data1[:,2]
sigVx = data1[:,13]

Noli = Noli*5/(50*100*100 + Noli*5)

Nw = 766*1e3

print(F.shape)


a = 0; b = a + 8; i=0
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax1.set_ylim([60,80])

ax.text(0.0525, 0.1, '(a)', fontsize=12)

ax.set_xlabel(r"$v_x~[r_c/\tau]$")
ax.set_ylabel(r"$P_{diss}~[10^{-5}~k_BT/\tau]$")

ax.legend()

plt.savefig("figures/PdissvsV.pdf")

# input("press enter")

ax.cla()


a = 0; b = a + 8; i=0
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, linestyle='none', marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, linestyle='none', marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, linestyle='none', marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],vx[a:b]*F[a:b]*Nw, yerr=sigVx[a:b]*F[a:b]*Nw, linestyle='none', marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax1.set_ylim([60,80])

ax.text(3.5e-2, 0.6e-8, '(b)', fontsize=12)

ax.set_xlabel(r"$v_x~[r_c/\tau]$")
ax.set_ylabel(r"$P_{diss}~[k_BT/\tau]$")

ax.legend()

vxmin = vx.min()
vxmax = vx.max()

a = 0; b = a + 8; i=0

N = len(F[a:b])
xx = np.log(vx[a:b])
yy = np.log(vx[a:b]*F[a:b]*Nw)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)


ax.plot([vxmin,vxmax],[np.exp(p1[1])*vxmin**p1[0],np.exp(p1[1])*vxmax**p1[0]], linestyle= '--', color=col[i])


a = b; b = a + 8; i=i+1

N = len(F[a:b])
xx = np.log(vx[a:b])
yy = np.log(vx[a:b]*F[a:b]*Nw)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)

ax.plot([vxmin,vxmax],[np.exp(p1[1])*vxmin**p1[0],np.exp(p1[1])*vxmax**p1[0]], linestyle= '--', color=col[i])




a = b; b = a + 8; i=i+1

N = len(F[a:b])
xx = np.log(vx[a:b])
yy = np.log(vx[a:b]*F[a:b]*Nw)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)

ax.plot([vxmin,vxmax],[np.exp(p1[1])*vxmin**p1[0],np.exp(p1[1])*vxmax**p1[0]], linestyle= '--', color=col[i])




a = b; b = a + 8; i=i+1

N = len(F[a:b])
xx = np.log(vx[a:b])
yy = np.log(vx[a:b]*F[a:b]*Nw)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)

ax.plot([vxmin,vxmax],[np.exp(p1[1])*vxmin**p1[0],np.exp(p1[1])*vxmax**p1[0]], linestyle= '--', color=col[i])


ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig("figures/PdissvsV_loglog.pdf")

input("press enter")
