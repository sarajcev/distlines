import numpy as np
import matplotlib.pyplot as plt

from sandbox import amplitude_distance_bivariate_pdf
from sandbox import risk_from_clp, risk_from_clp_points
from distlines import critical_current, critical_current_chowdhuri
from distlines import critical_current_fit, ieee_std_1410
from utils import moving_window


# Figure style using matplotlib:
fig_style = 'seaborn-v0_8-colorblind'
plt.style.use(fig_style)

# Number of random samples:
N = 1500
# Distribution line geometry (single line example):
Un = 20.    # nominal voltage (kV)
h = 11.5    # shield wire height (m)
y = 10.     # phase conductor height (m)
sg = 3.     # distance between shield wires (m)
CFO = 150.  # critical flashover voltage (kV)
# Tower's grounding system.
grounding_type = 'P'  # ring-type
length_type = '1&5'   # 5 m length
r_tower = 1.
rho_soil = 100.
# Ground flash density:
Ng = 1.  # strikes/(km^2*year)
risk_factor = 0.2*Ng  # per 100-km/years
# Lightning-current parameters.
lgtn_params = 'default'
if lgtn_params == 'default':
    muI = 31.1
    sigmaI = 0.484
elif lgtn_params == 'alternative':
    muI = 34.0
    sigmaI = 0.740

# Domain extent.
XMAX = 500.
YMAX = 200.

print('Running ...')

# Compute number of flashovers per 100 km per year using the
# method from the IEEE Std. 1410 (for different CFO values).
cfos = np.arange(50, 301, 1)  # see original paper
nfsh = np.empty_like(cfos)
nfno = np.empty_like(cfos)
for i in range(len(cfos)):
    # Shield wires present.
    nfsh[i] = ieee_std_1410(h, y, sg, cfos[i], shield=True)
    # Shield wires absent.
    nfno[i] = ieee_std_1410(h, y, sg, cfos[i], shield=False)

fig, ax = plt.subplots(figsize=(5.5,4))
ax.set_title('Ng = 1 km^2/year', fontweight='bold', fontsize=10)
ax.semilogy(cfos, nfno, ls='-', lw=2, label='w/o shield wires')
ax.semilogy(cfos, nfsh, ls='--', lw=2, label='with shield wires')
ax.legend(loc='upper right')
ax.set_xlabel('CFO (kV)', fontweight='bold', fontsize=10)
ax.set_ylabel('No. flashovers per 100 km year', fontweight='bold', fontsize=10)
ax.grid()
fig.tight_layout()
plt.savefig('ieee_1410.png', dpi=600)
plt.show()

# Compute critical currents of the line by the deterministic
# method, both with and without the shield wire(s). This is
# based on the Rusck's model per IEEE Std. 1410.
ds = np.arange(1, XMAX)
cc0 = np.empty_like(ds)
cc1 = np.empty_like(ds)
cc0v = np.empty_like(ds)
cc1v = np.empty_like(ds)
cc2 = np.empty_like(ds)
cc3 = np.empty_like(ds)

for i in range(len(ds)):
    # Default case.
    # Without shield wire(s).
    cc0[i] = critical_current(ds[i], y, h, 0, sg, 100., CFO, k_cfo=1.5)
    # With shield wire(s).
    cc1[i] = critical_current(ds[i], y, h, 1, sg, 100., CFO, k_cfo=1.5)
    # Lightning return-stroke velocity.
    # Without shield wire(s).
    cc0v[i] = critical_current(ds[i], y, h, 0, sg, 150., CFO, k_cfo=1.5)
    # With shield wire(s).
    cc1v[i] = critical_current(ds[i], y, h, 1, sg, 150., CFO, k_cfo=1.5)
    # CFO = 200 kV.
    # Without shield wire(s).
    cc2[i] = critical_current(ds[i], y, h, 0, sg, 100., 200., k_cfo=1.5)
    # With shield wire(s).
    cc3[i] = critical_current(ds[i], y, h, 1, sg, 100., 200., k_cfo=1.5)

# Fit the polynomial to the critical currents.
# Default case.
clp0 = critical_current_fit(ds, cc0)
clp1 = critical_current_fit(ds, cc1)
# Lightning return-stroke velocity
clp0v = critical_current_fit(ds, cc0v)
clp1v = critical_current_fit(ds, cc1v)
# CFO = 200 kV
clp2 = critical_current_fit(ds, cc2)
clp3 = critical_current_fit(ds, cc3)

# Compute the risk from a double integral under the
# bivariate PDF of lightning currents and distances.
risk0 = risk_from_clp(clp0, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk w/o  shield wire: {risk0:.4f}')
risk1 = risk_from_clp(clp1, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk with shield wire: {risk1:.4f}')
# Lightning return-stroke velocity
risk0v = risk_from_clp(clp0v, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk w/o  shield wire: {risk0v:.4f}')
risk1v = risk_from_clp(clp1v, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk with shield wire: {risk1v:.4f}')
# CFO = 200 kV
risk2 = risk_from_clp(clp2, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk w/o  shield wire: {risk2:.4f}')
risk3 = risk_from_clp(clp3, 0., XMAX, mu=muI, sigma=sigmaI)
print(f'Risk with shield wire: {risk3:.4f}')

# Compute critical currents of the line by the deterministic
# method, both with and without the shield wire(s), based on
# the Chowdhuri-Gross model.
cc0r = np.empty_like(ds)
cc1r = np.empty_like(ds)
cc0s = np.empty_like(ds)
for i in range(len(ds)):
    print(f'- {i+1} / {len(ds)} ...')
    # Wavefront duration of 1 us.
    # Without shield wire(s).
    cc0r[i] = critical_current_chowdhuri(ds[i], 1., y, h, 0., sg, CFO)
    # With shield wire(s).
    cc1r[i] = critical_current_chowdhuri(ds[i], 1., y, h, 1., sg, CFO)
    # Wavefront duration of 2 us.
    # Without shield wire(s).
    cc0s[i] = critical_current_chowdhuri(ds[i], 2., y, h, 0., sg, CFO)

# Apply moving window on the CLP curves.
y_cc0r = moving_window(cc0r, window='blackman', N=50)
y_cc1r = moving_window(cc1r, window='blackman', N=50)
y_cc0s = moving_window(cc0s, window='blackman', N=50)

# Compute the risk.
print('Chowdhuri-Gross model:')
risk0r = risk_from_clp_points(ds[::2], y_cc0r[::2], mu=muI, sigma=sigmaI)
print(f'Risk at 1 us w/o  shield wire: {risk0r:.4f}')
risk1r = risk_from_clp_points(ds[::2], y_cc1r[::2], mu=muI, sigma=sigmaI)
print(f'Risk at 1 us with shield wire: {risk1r:.4f}')
risk0s = risk_from_clp_points(ds[::2], y_cc0s[::2], mu=muI, sigma=sigmaI)
print(f'Risk at 2 us w/o  shield wire: {risk0s:.4f}')

# Plot different CLP curves (1/3).
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.set_title('CFO = 150 kV',fontweight='bold', fontsize=10)
ax.plot(ds, cc0, ls='-', lw=1.5, label='v = 100 m/us (w/o shield)')
ax.plot(ds, cc1, ls='-', lw=1.5, label='v = 100 m/us (with shield)')
ax.plot(ds, cc0v, ls='--', lw=2, label='v = 150 m/us (w/o shield)')
ax.plot(ds, cc1v, ls='--', lw=2, label='v = 150 m/us (with shield)')
ax.legend(loc='best')
ax.set_xlabel('Distance (m)', fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (kA)', fontweight='bold', fontsize=10)
ax.grid(which='major', axis='both')
ax.set_ylim(0, 300)
fig.tight_layout()
plt.savefig('clp1.png', dpi=600)
plt.show()

# Plot different CLP curves (2/3).
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.set_title('Return-stroke velocity of 100 m/us',
             fontweight='bold', fontsize=10)
ax.plot(ds, cc0, ls='-', lw=1.5, label='CFO = 150 kV (w/o shield)')
ax.plot(ds, cc1, ls='-', lw=1.5, label='CFO = 150 kV (with shield)')
ax.plot(ds, cc2, ls='--', lw=2, label='CFO = 200 kV (w/o shield)')
ax.plot(ds, cc3, ls='--', lw=2, label='CFO = 200 kV (with shield)')
ax.legend(loc='best')
ax.set_xlabel('Distance (m)', fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (kA)', fontweight='bold', fontsize=10)
ax.grid(which='major', axis='both')
ax.set_ylim(0, 300)
fig.tight_layout()
plt.savefig('clp2.png', dpi=600)
plt.show()

# Plot different CLP curves (3/3).
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.set_title('Chowdhuri-Gross with CFO = 150 kV', 
             fontweight='bold', fontsize=10)
ax.plot(ds, y_cc0r, ls='-', lw=1.5, label='tf = 1 us (w/o shield)')
ax.plot(ds, y_cc1r, ls='-', lw=1.5, label='tf = 1 us (with shield)')
ax.plot(ds, y_cc0s, ls='-', lw=1.5, label='tf = 2 us (w/o shield)')
ax.legend(loc='best')
ax.set_xlabel('Distance (m)', fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (kA)', fontweight='bold', fontsize=10)
ax.grid(which='major', axis='both')
ax.set_ylim(0, 300)
fig.tight_layout()
plt.savefig('clp3.png', dpi=600)
plt.show()

# Prepare meshgrid for prediction.
xx = np.linspace(0, XMAX, 200)    # x-axis (distance)
yy = np.linspace(0, YMAX, 200).T  # y-axis (amplitude)
xx, yy = np.meshgrid(xx, yy)
# Bivariate PDF of lightning-current amplitudes and distances.
args = (0., XMAX, muI, sigmaI)
zz_pdf = amplitude_distance_bivariate_pdf(yy, xx, *args)

# Graphical visualization in the 3D.
offset = -1e-6
fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(projection='3d')
ax.set_title('CFO = 150 kV & v = 100 m/us',fontweight='bold', fontsize=10)
# Bivariate probability density function (of distances and amplitudes).
ax.plot_surface(xx, yy, zz_pdf, edgecolor='royalblue', lw=0.5,
                rstride=8, cstride=16, alpha=0.2)
# Show select CLP curves.
cc0 = cc0[np.logical_and(cc0>0, cc0<YMAX)]
cc1 = cc1[np.logical_and(cc1>0, cc1<YMAX)]
ax.plot(ds[:len(cc0)], cc0, ls='--', lw=2, c='blueviolet', label='w/o shield')
ax.plot(ds[:len(cc1)], cc1, ls='-', lw=2, c='magenta', label='with shield')
# Projection of the PDF surface on the figure floor.
ax.contourf(xx, yy, zz_pdf, zdir='z', offset=offset, cmap='viridis', alpha=0.5)
# Axis labels and limits.
ax.set_xlabel('Distance (m)', fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (kA)', fontweight='bold', fontsize=10)
ax.set_zlabel('Probability', fontweight='bold', fontsize=10)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.legend(loc='best')
fig.tight_layout()
plt.savefig('3d-'+lgtn_params+'.png', dpi=600)
plt.show()

