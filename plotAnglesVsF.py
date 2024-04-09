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
fig2, ax2 = plt.subplots(num=102)
fig3, ax3 = plt.subplots(num=103)
fig4, ax4 = plt.subplots(num=104)
fig5, ax5 = plt.subplots(num=105)
fig6, ax6 = plt.subplots(num=106)
fig7, ax7 = plt.subplots(num=107)
fig8, ax8 = plt.subplots(num=108)
fig9, ax9 = plt.subplots(num=109)


data1 = np.genfromtxt('mean_Noli_F_vx_adv_rec_semiA_semiB_rotTh_subsH_ridgeHL_ridgeHR_brushHL_brushHR', delimiter=',')

# print(data1)

ii = 0

Noli    = data1[:,ii]
ii = ii+1
F   = data1[:,ii]
ii = ii+1
vx = data1[:,ii]
ii = ii+1
adv   = data1[:,ii]
ii = ii+1
rec   = data1[:,ii]
ii = ii+1
semiA   = data1[:,ii]
ii = ii+1
semiB   = data1[:,ii]
ii = ii+1
rotTh   = data1[:,ii]
ii = ii+1
substH = data1[:,ii]
ii = ii+1
ridgeHL = data1[:,ii]
ii = ii+1
ridgeHR = data1[:,ii]
ii = ii+1
brushHL = data1[:,ii]
ii = ii+1
brushHR = data1[:,ii]

ii = ii+1
sigvx   = data1[:,ii]
ii = ii+1
sigadv   = data1[:,ii]
ii = ii+1
sigrec   = data1[:,ii]
ii = ii+1
sigsemiA   = data1[:,ii]
ii = ii+1
sigsemiB   = data1[:,ii]
ii = ii+1
sigrotTh   = data1[:,ii]
ii = ii+1
sigsubstH = data1[:,ii]
ii = ii+1
sigridgeHL = data1[:,ii]
ii = ii+1
sigridgeHR = data1[:,ii]
ii = ii+1
sigbrushHL = data1[:,ii]
ii = ii+1
sigbrushHR = data1[:,ii]

Noli = Noli*5/(50*100*100 + Noli*5)

ecc = np.sqrt(1-semiB**2/semiA**2)

sigEcc = np.sqrt( (semiB/ecc/semiA**2*sigsemiB)**2 + (semiB**2/semiA**3/ecc*sigsemiA)**2 )

sepHR = ridgeHR - brushHR

sigSepHR = np.sqrt(sigridgeHR**2 + sigbrushHR**2)

sepHL = ridgeHL - brushHL

sigSepHL = np.sqrt(sigridgeHL**2 + sigbrushHL**2)

print(sigSepHR)


a = 0; b = a + 8; i=0
ax.errorbar(vx[a:b],adv[a:b], yerr=sigadv[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],adv[a:b], yerr=sigadv[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],adv[a:b], yerr=sigadv[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax.errorbar(vx[a:b],adv[a:b], yerr=sigadv[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

ax.set_xlabel(r"$v_x~[r_c/\tau]$")
ax.set_ylabel(r"$\theta_{adv}^{app}~[deg]$")

ax.legend()


ax.text(0.055, 96, '(a)', fontsize=12)


fig.savefig("figures/advVSF.pdf")

a = 0; b = a + 8; i=0
ax1.errorbar(vx[a:b],rec[a:b], yerr=sigrec[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax1.errorbar(vx[a:b],rec[a:b], yerr=sigrec[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax1.errorbar(vx[a:b],rec[a:b], yerr=sigrec[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax1.errorbar(vx[a:b],rec[a:b], yerr=sigrec[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax1.set_ylim([60,80])

ax1.set_xlabel(r"$v_x~[r_c/\tau]$")
ax1.set_ylabel(r"$\theta_{rec}^{app}~[deg]$")

ax1.legend()


ax1.text(0.055, 72, '(b)', fontsize=12)


fig1.savefig("figures/recVSF.pdf")

a = 0; b = a + 8; i=0
ax2.errorbar(vx[a:b],ecc[a:b], yerr=sigEcc[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax2.errorbar(vx[a:b],ecc[a:b], yerr=sigEcc[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax2.errorbar(vx[a:b],ecc[a:b], yerr=sigEcc[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax2.errorbar(vx[a:b],ecc[a:b], yerr=sigEcc[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )


ax2.set_xlabel(r"$v_x~[r_c/\tau]$")
ax2.set_ylabel(r"$eccentricity$")

ax2.legend()

ax2.text(0.055, 0.45, '(c)', fontsize=12)

fig2.savefig("figures/eccVSF.pdf")


a = 8; b = a + 8; i=0
# ax3.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$Liquid:\Phi=${:.3g}'.format(Noli[a]) )
ax3.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i+1], color=col[i+1], label='$Brush:\Phi=${:.3g}'.format(Noli[a]) )


ax3.set_xlabel(r"$v_x~[r_c/\tau]$")
ax3.set_ylabel(r"$h~[r_c]$")

# ax3.set_ylim([10,14])

ax3.legend()

fig3.savefig("figures/ridgeHvsV_{:.3g}.pdf".format(Noli[a]))

a = b; b = a + 8; i=0
# ax4.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$Liquid:\Phi=${:.3g}'.format(Noli[a]) )
ax4.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i+1], color=col[i+1], label='$Brush:\Phi=${:.3g}'.format(Noli[a]) )


ax4.set_xlabel(r"$v_x~[r_c/\tau]$")
ax4.set_ylabel(r"$h~[r_c]$")

# ax4.set_ylim([10,14])

ax4.legend()

fig4.savefig("figures/ridgeHvsV_{:.3g}.pdf".format(Noli[a]))

a = b; b = a + 8; i=0
# ax5.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$Liquid:\Phi=${:.3g}'.format(Noli[a]) )
ax5.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i+1], color=col[i+1], label='$Brush:\Phi=${:.3g}'.format(Noli[a]) )


ax5.set_xlabel(r"$v_x~[r_c/\tau]$")
ax5.set_ylabel(r"$h~[r_c]$")

# ax5.set_ylim([10,14])

ax5.legend()

fig5.savefig("figures/ridgeHvsV_{:.3g}.pdf".format(Noli[a]))


a = 0; b = a + 8; i=0
# ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
# ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

ax6.set_xlabel(r"$v_x~[r_c/\tau]$")
ax6.set_ylabel(r"$h_{sep}^{adv}~[r_c]$")

ax6.legend()

ax6.text(0.0275, 15, '(c)', fontsize=12)

fig6.savefig("figures/sepHvsV_adv.pdf")


a = 0; b = a + 8; i=0
# ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
# ax6.errorbar(vx[a:b],sepHR[a:b], yerr=sigSepHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax7.errorbar(vx[a:b],sepHL[a:b], yerr=sigSepHL[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax7.errorbar(vx[a:b],sepHL[a:b], yerr=sigSepHL[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax7.set_ylim([-1,17])

ax7.set_xlabel(r"$v_x~[r_c/\tau]$")
ax7.set_ylabel(r"$h_{sep}^{rec}~[r_c]$")

ax7.legend()

ax7.text(0.0275, 15, '(d)', fontsize=12)

fig7.savefig("figures/sepHvsV_rec.pdf")


a = 0; b = a + 8; i=0
ax8.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax8.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax8.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax8.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax8.set_yscale('log')

ax8.set_xlabel(r"$v_x~[r_c/\tau]$")
ax8.set_ylabel(r"$h_{B}^{adv}~[r_c]$")

# ax8.set_ylim([7,13.9])

ax8.legend(loc="lower right")

ax8.text(0.0275, 13.25, '(a)', fontsize=12)

fig8.savefig("figures/brushHvsV.pdf")


a = 0; b = a + 8; i=0
# ax9.errorbar(vx[a:b],brushHR[a:b], yerr=sigbrushHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax9.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax9.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )
a = b; b = a + 8; i=i+1
ax9.errorbar(vx[a:b],ridgeHR[a:b], yerr=sigridgeHR[a:b], marker=mar[i], color=col[i], label='$\Phi=${:.3g}'.format(Noli[a]) )

# ax9.set_yscale('log')

ax9.set_xlabel(r"$v_x~[r_c/\tau]$")
ax9.set_ylabel(r"$h_{R}^{adv}~[r_c]$")

ax9.legend()

ax9.text(0.0275, 28, '(b)', fontsize=12)

fig9.savefig("figures/ridgeHvsV.pdf")

input("press enter")
