import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource

# ---- parameters ----
w_R = 0.6
w_AO = 0.4
R_thr_km = 20.0
p30 = 0.1  # f_R(30km)=p30

# ---- definitions ----
def g_AO(AO):
    """AO in radians -> [0,1]"""
    return 0.5 * (1.0 + np.cos(AO))

def f_R_logistic(R_km, R_thr_km=20.0, p30=0.1):
    """logistic distance reward"""
    beta = np.log((1 - p30) / p30) / (30.0 - R_thr_km)
    return 1.0 / (1.0 + np.exp(beta * (R_km - R_thr_km)))

def phi(R_km, AO_rad):
    return w_R * f_R_logistic(R_km, R_thr_km=R_thr_km, p30=p30) + w_AO * g_AO(AO_rad)

# ---- grid ----
R_vals = np.linspace(0, 35, 220)     # km
AO_vals = np.linspace(0, np.pi, 180) # rad
R_grid, AO_grid = np.meshgrid(R_vals, AO_vals)
Phi = phi(R_grid, AO_grid)

# ---- colormap + lighting ----
cmap = cm.plasma  # vibrant gradient
norm = colors.Normalize(vmin=Phi.min(), vmax=Phi.max())
ls = LightSource(azdeg=315, altdeg=45)
shaded_rgb = ls.shade(Phi, cmap=cmap, norm=norm, vert_exag=0.8, blend_mode='soft')

# ---- 3D surface ----
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    R_grid, AO_grid * 180/np.pi, Phi,
    facecolors=shaded_rgb,
    rstride=2, cstride=2, antialiased=True, linewidth=0
)

# ground contour for extra "depth" effect
ax.contour(
    R_grid, AO_grid * 180/np.pi, Phi,
    zdir='z', offset=Phi.min() - 0.05, levels=18, cmap=cmap, linewidths=0.6
)

# labels and view
ax.set_xlabel("Distance R (km)")
ax.set_ylabel("AO (degrees)")
ax.set_zlabel("Φ")
ax.set_title("Stylized 3D Surface of Φ(R, AO)")

# colorbar mapped to Phi
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(Phi)
fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Φ")

ax.view_init(elev=30, azim=-55)
plt.tight_layout()
out_path = "/mnt/data/phi_surface_cool.png"
# plt.savefig(out_path, dpi=180)
plt.show()

# out_path
