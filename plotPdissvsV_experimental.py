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
col=['#cc0000','#0000cc','#009900','#ff8000','#6f00ff','#e500e5','#e5c100','#858585','#000000']


plt.ion();


fig, ax = plt.subplots(num=100)


data1 = np.genfromtxt('cvd_brushes_v_F_sigF', delimiter=',')
data2 = np.genfromtxt('dropCast Brushes_v_F_sigF', delimiter=',')

v1    = data1[1,1:]
F1    = data1[2,1:]
sigF1 = data1[3,1:]

v2    = data2[1,1:]
F2    = data2[2,1:]
sigF2 = data2[3,1:]


i=0
ax.errorbar(v1,F1*v1, yerr=sigF1*v1, linestyle= 'none', marker=mar[i], color=col[i], label='grafted from' )
i=1
ax.errorbar(v2,F2*v2, yerr=sigF2*v2, linestyle= 'none', marker=mar[i], color=col[i], label='grafted to' )

ax.text(8, 2.7e-7, '(b)', fontsize=12)

Vmin = v1.min()
Vmax = v1.max()

i=0

N = len(F1)
xx = np.log(v1)
yy = np.log(F1*v1)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)

ax.plot([Vmin,Vmax],[np.exp(p1[1])*Vmin**p1[0],np.exp(p1[1])*Vmax**p1[0]], linestyle= '--', color=col[i])

print("slope=", p1[0])

Vmin = v2.min()
Vmax = v2.max()


i=1

N = len(F2)
xx = np.log(v2)
yy = np.log(F2*v2)
delta = N*(xx**2).sum() - ( xx.sum() )**2
p1 = np.polyfit(xx,yy,1)
A = p1[1]
B = p1[0]

ei2 = ( (yy - A - B*xx)**2 ).sum()

sigB = np.sqrt( N/(N-2)*ei2/delta )

p1 = np.polyfit(xx,yy,1)

print("slope=", B)
print("err=", sigB)

ax.plot([Vmin,Vmax],[np.exp(p1[1])*Vmin**p1[0],np.exp(p1[1])*Vmax**p1[0]], linestyle= '--', color=col[i])

print("slope=", p1[0])

ax.set_xlabel(r"$v$ [mm/s]")
ax.set_ylabel(r"$P_{diss}$ [mW]")


ax.set_xscale('log')
ax.set_yscale('log')


ax.legend()

plt.savefig("figures/PdissvsV_exp_loglog.pdf")

input("press enter")
