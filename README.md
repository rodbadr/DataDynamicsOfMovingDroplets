Processed data and plotting codes for the published article:

R.G.M. Badr, L. Hauer, D. Vollmer, and F. Schmid, Dynamics of Droplets Moving on Lubricated Polymer Brushes. Langmuir 2024, 40, 24, 12368–12380. DOI: https://doi.org/10.1021/acs.langmuir.4c00400


Data is in comma separated value (csv) format. Below is a description of every data file. The codes are python codes that read the data files and generate the plots that appear in the publication.

# Data files

## cvd_brushes_v_F_sigF.csv

Experimental measurement of force vs velocity for brushes synthesized through chemical vapor deposition (cvd).

Row 1 and column 1 are for indexing.
Row 2 is the velocity of the stage.
Row 3 is the average measured force.
Row 4 is the standard error on the force measurement.

## dropCast_brushes_v_F_sigF.csv

Experimental measurement of force vs velocity for brushes synthesized through drop casting (dropCast).

Row 1 and column 1 are for indexing.
Row 2 is the velocity of the stage.
Row 3 is the average measured force.
Row 4 is the standard error on the force measurement.

## mean_Noli_F_vx_adv_rec_semiA_semiB_rotTh_subsH_ridgeHL_ridgeHR_brushHL_brushHR.csv

Simulation results for different quantities, varying the lubrication fraction and applied force.

Every row corresponds to a parameter set.

Every column corresponds to the quantity referred to in the filename in corresponding order, followed by the standard error of the mean (SEM) on every measured value (all values except Noli and F)

Key:

- Noli: number of lubricant chains
- F: magnitude of the applied external force in the x-direction
- vx: Measured steady state center of mass velocity of the droplet
- adv: advancing contact angle
- rec: receding contact angle
- semiA: semi major axis of elliptical fit of the droplet
- semiB: semi minor axis of elliptical fit of the droplet
- rotTh: rotation angle of the elliptical fit, measured from the positive x-axis
- substH: height of the unperturbed brush-lubricant far away from the droplet
- ridgeHL: height of the full ridge at the receding end
- ridgeHR: height of the full ridge at the advancing front
- brushHL: height of the brush chains in the ridge at the receding end
- brushHR: height of the brush chains in the ridge at the advancing front

## mean_Noli_F_vx_surfDensTop.csv

Simulation results for the effective surface density at the top of the droplet, varying the lubrication fraction and applied force.


Every row corresponds to a parameter set.

Every column corresponds to the quantity referred to in the filename in corresponding order, followed by the standard error of the mean (SEM) on the surface density.

Key:

- Noli: number of lubricant chains
- F: magnitude of the applied external force in the x-direction
- vx: Measured steady state center of mass velocity of the droplet
- surfDensTop: effective surface density at the top of the droplet


# Plotting codes

The following names which codes generate which figure in the article.

## Figure 2

plotFvsV_experimental.py
plotPdissvsV_experimental.py



## Figure 3

plotFvsV.py

## Figure 4

plotPdissvsV.py

## Figures 6 and 7

plotAnglesVsF.py

## Figure 12 (b)

plotSurfDensTopvsV.py


