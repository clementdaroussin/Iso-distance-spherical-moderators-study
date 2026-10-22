import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

###########################################################################################
###########################################################################################
############################## SENSITIVITY ANALYSIS #######################################
###########################################################################################
###########################################################################################
# Data
radii = np.array([50, 75, 100, 125, 150, 175, 200])
# 1e-5 pour représenter le "0" en log
leakage_h2o = np.array([0.24, 0.02, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]) 
leakage_graphite = np.array([94.09, 80.66, 64.80, 50.17, 37.38, 27.22, 19.32])
leakage_d2o = np.array([98.82, 96.62, 93.72, 90.08, 86.11, 81.74, 76.89])

plt.figure(figsize=(10, 6))

# Plotting lines
plt.plot(radii, leakage_h2o, 'o-', label='Light Water (H2O)', color='royalblue', lw=2)
plt.plot(radii, leakage_graphite, 's-', label='Graphite', color='dimgray', lw=2)
plt.plot(radii, leakage_d2o, '^-', label='Heavy Water (D2O)', color='lightseagreen', lw=2)
plt.axvline(x=125, color='red', linestyle='--', lw=1, label='Analysis Threshold (125 cm)')

# Formatting
plt.xlabel('Sphere Radius (cm)', fontsize=12)
plt.ylabel('Neutron Leakage (%)', fontsize=12)
plt.title('Neutron Leakage vs. Shielding Thickness', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# Adding an annotation for the "0%" Water
plt.annotate('Total Shielding', xy=(100, 1e-5), xytext=(70, 20),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.savefig('./res/1.Leakage_vs_Shielding_Thickness.png', dpi=300)

#plt.show()
###########################################################################################
###########################################################################################
################################# POST-PROCESSING #########################################
###########################################################################################
###########################################################################################

###########################################################################################
############################### MC PARAMETERS #############################################
###########################################################################################

# Study Parameters
sphere_radius = 125 # in cm
nb_particles = 1000000
nb_batches = 10
mesh_size = 300

###########################################################################################
################################### MATERIALS #############################################
###########################################################################################

mats = openmc.Materials()

# 1. Water (H2O)
water = openmc.Material(material_id=1, name="light_water")
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O') 

# 2. Heavy Water (D2O)
heavy_water = openmc.Material(material_id=2, name="heavy_water")
heavy_water.add_nuclide('H2', 2.0)
heavy_water.add_nuclide('O16', 1.0)
heavy_water.set_density('g/cm3', 1.1) # Densité plus élevée que H2O
heavy_water.add_s_alpha_beta('c_D_in_D2O') 

# 3. Graphite (Nuclear Grade)
graphite = openmc.Material(material_id=3, name="graphite")
graphite.add_element('C', 1.0)
graphite.set_density('g/cm3', 1.7) # Densité typique du graphite nucléaire
graphite.add_s_alpha_beta('c_Graphite') 


# adding and exporting material
mats.append(water)
mats.append(heavy_water)
mats.append(graphite)

mats.export_to_xml()

###########################################################################################
################################### GEOMETRY ##############################################
###########################################################################################

# Surfaces
external_surface = openmc.Sphere(r=sphere_radius, boundary_type='vacuum')

# Cells
cell = openmc.Cell(fill=water, region=-external_surface)

#universe
univ = openmc.Universe(cells=[cell])

#final geometry
geom = openmc.Geometry(univ)
geom.export_to_xml()

###########################################################################################
################################### VISUALIZATION #########################################
###########################################################################################

fig, axes_list = plt.subplots(1, 3, figsize=(18, 6))

plot_configs = [
    (water, 'Light Water', 'royalblue'),
    (heavy_water, 'Heavy Water', 'lightseagreen'),
    (graphite, 'Graphite', 'dimgray')]

for i, (mat, name, color_code) in enumerate(plot_configs):
    ax = axes_list[i]
    cell.fill = mat
    
    # Visualizing the geometry slice
    univ.plot(width=(sphere_radius * 2.2, sphere_radius * 2.2), 
              basis='xy', 
              color_by='material', 
              colors={mat: color_code},
              axes=ax,
              pixels=1000000)
    
    # Marking the point source at the center
    ax.scatter(0, 0, color='red', s=20, label='Source', zorder=5)
    
    # Adding radius annotation for scale clarity
    ax.annotate('', xy=(sphere_radius*0.98, 0), xytext=(sphere_radius*0.02, 0), 
    	arrowprops=dict(arrowstyle='<->', color='Black', lw=1.5))
    ax.text(sphere_radius/2, 8, f'R = {sphere_radius} cm', 
           	color='Black', fontsize=10, ha='center', fontweight='bold')

    ax.set_title(f"{name}")
    
plt.tight_layout()
plt.savefig('./res/2.Geometry_Visualization.png', dpi=300)
#plt.show()

###########################################################################################
################################### SOURCES ###############################################
###########################################################################################

source = openmc.IndependentSource()
source.space = openmc.stats.Point((0, 0, 0))
source.energy = openmc.stats.Watt()

###########################################################################################
################################### SETTINGS ##############################################
###########################################################################################

sett = openmc.Settings()
sett.source = source
sett.particles = nb_particles
sett.batches = nb_batches
sett.run_mode = 'fixed source'

sett.export_to_xml()

###########################################################################################
################################### MESH ##################################################
###########################################################################################

# Create mesh
mesh = openmc.RegularMesh()
mesh.dimension = [mesh_size, mesh_size, 1]
mesh.lower_left = [-sphere_radius, -sphere_radius, -1]
mesh.upper_right = [sphere_radius, sphere_radius, 1]

# Create filter (average in every mesh not only a global one)
mesh_filter = openmc.MeshFilter(mesh)

###########################################################################################
################################### TALLIES ###############################################
###########################################################################################

tallies = openmc.Tallies()

# 1. Energy Spectrum Tally
energies = np.logspace(-5, 7, 501)
e_filter = openmc.EnergyFilter(energies)
spec_tally = openmc.Tally(name='flux_spectrum')
spec_tally.filters = [e_filter]
spec_tally.scores = ['flux']
tallies.append(spec_tally)

# 2. Spatial Flux Tally
mesh_tally = openmc.Tally(name='spatial_flux')
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = ['flux']
tallies.append(mesh_tally)

# 3. Leakage Tally
# Requires a surface filter on the outer boundary of your sphere
surf_filter = openmc.SurfaceFilter([external_surface]) 
leak_tally = openmc.Tally(name='leakage_current')
leak_tally.filters = [surf_filter]
leak_tally.scores = ['current'] 
tallies.append(leak_tally)

# 4. Moderation Time Tally
# Tracks how long neutrons stay in the system
times = np.logspace(-9, -2, 101) # From nano-s to centi-s
t_filter = openmc.TimeFilter(times)
time_tally = openmc.Tally(name='moderation_time')
time_tally.filters = [t_filter]
time_tally.scores = ['flux']
tallies.append(time_tally)

# 5. Absorption Rate Tally
# This tracks the physical disappearance of neutrons by capture/absorption
abs_tally = openmc.Tally(name='absorption_rate')
abs_tally.scores = ['absorption']
tallies.append(abs_tally)

tallies.export_to_xml()

###########################################################################################
################################### AUTOMATED RUN #########################################
###########################################################################################

material_list = [(water, 'light_water'), (graphite, 'graphite'), (heavy_water, 'heavy_water')]

# Initialize data containers for post-processing
results_2d = {}
results_spectrum = {}
results_time = {}      
results_leakage = {}   
results_abs = {}

for mat_obj, name in material_list:
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"{name.upper()}")
    print(f"Configuration: {nb_particles} particles | {nb_batches} batches")
    print(f"{'='*60}")
    
    # Update geometry and export
    cell.fill = mat_obj
    geom.export_to_xml()
    
    openmc.run(output=False)     
    
    # Access the statepoint file to harvest tally results
    statepoint_file = f'statepoint.{nb_batches}.h5'
    with openmc.StatePoint(statepoint_file) as sp:
        # 1. SPATIAL FLUX (2D)
        # Reshape the flat array into the spatial mesh grid
        raw_data = sp.get_tally(name='spatial_flux').get_values(scores=['flux'])
        results_2d[name] = raw_data.reshape((mesh_size, mesh_size)).copy()
        
        # 2. ENERGY SPECTRUM 
        # Capture the mean flux values across the defined energy bins
        spec_data = sp.get_tally(name='flux_spectrum').mean.flatten()
        results_spectrum[name] = spec_data.copy()
        # 3. MODERATION TIME 
        # Harvest the flux distribution as a function of neutron 'time-of-flight'
        time_data = sp.get_tally(name='moderation_time').mean.flatten()
        results_time[name] = time_data.copy()

        # 4. NEUTRON LEAKAGE AND ABSORPTION
        # Extract the global integral values for the final balance sheet
        results_leakage[name] = sp.get_tally(name='leakage_current').mean.sum()
        results_abs[name] = sp.get_tally(name='absorption_rate').mean.sum()
        
        elapsed_time = time.time() - start_time
        print(f"✅ {name.upper()} processed in {elapsed_time:.2f} s")
        
print('')
print("--- Simulation Sequence Complete, data Saved ---")

###########################################################################################
###########################################################################################
################################# POST-PROCESSING #########################################
###########################################################################################
###########################################################################################

###########################################################################################
##############################  SPECTRUM COMPARISON #######################################
###########################################################################################
# Create a figure with two vertical subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# Calculate geometric centers and bin widths
energy_centers = np.sqrt(energies[:-1] * energies[1:])
bin_widths = np.diff(energies)
for mat_obj, name in material_list:
    flux_vals = results_spectrum[name]
    # Calculate the standard flux density Phi(E)
    phi_e = flux_vals / bin_widths
    # 1. First plot: Standard Flux Phi(E)
    # Using log-log because Phi(E) spans several orders of magnitude
    ax1.loglog(energy_centers, phi_e, label=name, lw=1)
    # 2. Second plot: Lethargy Flux E * Phi(E)
    ax2.loglog(energy_centers, phi_e * energy_centers, label=name, lw=1)
    # Configuration for the first plot (Standard)
    ax1.set_ylabel(r'Flux Density [$\Phi(E)$]')
    ax1.set_title('Standard Energy Spectrum (1/E Slope)')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
ax1.legend()
# Configuration for the second plot (Lethargy)
ax2.set_xlabel('Energy [eV]')
12
ax2.set_ylabel(r'Lethargy Flux [$E \cdot \Phi(E)$]')
ax2.set_title('Lethargy-Normalized Spectrum (Horizontal Plateau)')
ax2.grid(True, which="both", ls="-", alpha=0.5)
# Thermal reference line on both plots
for ax in [ax1, ax2]:
    ax.axvline(x=0.0253, color='grey', linestyle='--', alpha=0.7, label='0.025 eV (Thermal)')
plt.tight_layout()
plt.savefig('./res/3.Spectrum_plotting_Comparison.png', dpi=300)
#plt.show()

###########################################################################################
################################ LETHARGY SPECTRUM ########################################
###########################################################################################

plt.figure(figsize=(15, 9))

# Geometric bin centers
energy_centers = np.sqrt(energies[:-1] * energies[1:])
bin_widths = np.diff(energies)
peaks=[]
#printing maxes tab
print(f"{'Material':<12} | {'E_peak (eV)':<12} | {'Flux_max'}")
for mat_obj, name in material_list:
    y = (results_spectrum[name] / bin_widths) * energy_centers
    idx = np.argmax(y[energy_centers < 1])
    peaks.append(y[idx])
    print(f"{name:<12} | {energy_centers[idx]:<12.4f} | {y[idx]:.2e}")
e_max_list = [energy_centers[np.argmax((results_spectrum[m[1]]/bin_widths)*energy_centers * 
(energy_centers < 1))] for m in material_list]
e_peak=np.mean(e_max_list)
print(f"Mean of peaks : {e_peak:.4f} eV")

i=0
for mat_obj, name in material_list:
    # Retrieve tally results for the specific material
    flux_vals = results_spectrum[name]
    # Flux density conversion
    plt.loglog(energy_centers, (flux_vals / bin_widths) * energy_centers, label=name, lw=1)
    plt.scatter(e_peak, peaks[i],marker='+', color='Black', s=30, zorder=5)
    plt.text(e_peak * 1.1, peaks[i], f"{peaks[i]:.3f} eV", 
     fontsize=10,verticalalignment='center')
    i+=1

# Thermal reference line
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.minorticks_on()
plt.xlabel('Energy [eV]')
plt.ylabel(r'Lethargy Flux [E(E)]')
plt.title('Energy Spectrum Comparison: Thermalization Profile')
plt.axvline(x=0.0253, color='grey', linestyle='--', alpha=0.7, label='0.025 eV (Thermal)')
plt.axvline(x=e_peak, color='red', linestyle='--', alpha=0.7, label='E of peaks')
plt.legend()
plt.savefig('./res/4.Lethargy_Spectrum.png', dpi=300)
#plt.show()

###########################################################################################
############################### TIME DISTRIBUTION #########################################
###########################################################################################

plt.figure(figsize=(15, 9))

# Loop on materials
for name in results_spectrum.keys():
    time_vals = results_time[name] 
    
    # Geometric centers
    time_centers = np.sqrt(times[:-1] * times[1:])
    
    # Log-log representation
    plt.loglog(time_centers, time_vals, label=name, lw=1)

plt.xlabel('Time [s]')
plt.ylabel('Flux Intensity')
plt.title('Neutron Slowing-Down Time Distribution')
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend()
plt.savefig('./res/5.Moderation_Time_Distribution.png', dpi=300)
#plt.show()

###########################################################################################
####################### 1D - RADIAL FLUX PROFILE (heatmap) ################################
###########################################################################################

# Calculate global limits to ensure all materials use the same color scale
all_data = np.concatenate([d.flatten() for d in results_2d.values()])
vmax = all_data.max()
vmin = vmax / 1e5  # Show 5 orders of magnitude

fig, axs = plt.subplots(len(results_2d), 1, figsize=(10, 2 * len(results_2d)))

if len(results_2d) == 1: axs = [axs]

for i, (name, data) in enumerate(results_2d.items()):
    # Extract radial slice from the horizontal center
    radial_slice = data[mesh_size // 2, :]
    band = radial_slice.reshape(1, -1)
    
    # Plot using global vmin/vmax for direct comparison
    im = axs[i].imshow(band, cmap='viridis', aspect='auto', 
                       extent=[-sphere_radius, sphere_radius, 0, 0.1],
                       norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    axs[i].set_title(f"Material: {name}")
    axs[i].set_yticks([]) 
    plt.colorbar(im, ax=axs[i], pad=0.01, label='Flux Intensity')

plt.xlabel('Distance from center [cm]')
plt.tight_layout()
plt.savefig('./res/6.Radial_Flux_Heatmap.png', dpi=300)
#plt.show()

###########################################################################################
########################## 1D RADIAL FLUX PROFILE #########################################
###########################################################################################

# Define material colors for consistent plotting
material_colors = {
    'light_water': 'royalblue',
    'heavy_water': 'lightseagreen',
    'graphite': 'dimgray'
    }
plt.figure(figsize=(15, 9))
print(f"{'Material':<15} | {'Slope':<10} | {'L (cm)':<10}")
print("-" * 40)
for name, data in results_2d.items():
    mid = mesh_size // 2
    radial_slice = data[mid, mid:]
    x_axis = np.linspace(0, sphere_radius, len(radial_slice))
    color = material_colors.get(name, 'black')
    # slope calculation
    fit_mask = (x_axis >= 5) & (x_axis <= 30)
    slope, _ = np.polyfit(x_axis[fit_mask], np.log(radial_slice[fit_mask] + 1e-15), 1)
    L = -1/slope

    # Forced Intercept
    intercept_max = np.log(radial_slice[0] + 1e-15)
    
    print(f"{name:<15} | {slope:<10.4f} | {L:.2f}")

    #Plotting
    plt.semilogy(x_axis, radial_slice, 
                 label=f'{name} (Slope={slope:<10.4f}; L={L:.2f})', 
                 color=color, lw=1.5)

    # Pure exponential dashed line
    plt.semilogy(x_axis, np.exp(slope * x_axis + intercept_max), 
                 color=color, ls='--', lw=1, alpha=0.4)
# Formatting
plt.xlabel('Distance from Center [cm]')
plt.ylabel('Neutron Flux (Log Scale)')
plt.title('1D Radial Flux Profile: Comparison with Pure Exponential Decay')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.ylim(1e-6, 1.0) # Keeps the plot clean
plt.xlim(0, 125) # Keeps the plot clean
plt.savefig('./res/7.Radial_Flux_Curve.png', dpi=300)
#plt.show()

###########################################################################################
########################### 2D SPATIAL FLUX DISTRIBUTION ##################################
###########################################################################################

fig, axes = plt.subplots(1, 3, figsize=(27, 9))

# Establish a common normalization scale for consistent visual comparison
vmax_common = max([res.max() for res in results_2d.values()]) 

# Loop on all materials
for i, (mat_obj, name) in enumerate(material_list):
    ax = axes[i]
    data_plot = results_2d[name]
    
    # Render spatial distribution using logarithmic scaling
    im = ax.imshow(data_plot, 
               extent=[-sphere_radius, sphere_radius, -sphere_radius, sphere_radius],
               norm=colors.LogNorm(vmin=1e-6, vmax=vmax_common), 
               cmap='viridis', 
               origin='lower',
               interpolation='none')
    
    # Overlay showing boundary of the geometry
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(sphere_radius*np.cos(theta), sphere_radius*np.sin(theta), color='red', 
    linewidth=1.5)
    
    # Configuration of limits and titles
    ax.set_xlim(-sphere_radius*1.05, sphere_radius*1.1)
    ax.set_ylim(-sphere_radius*1.05, sphere_radius*1.1)
    ax.set_title(f'{name}', fontsize=25)
    
    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Flux [n/cm²-s]')

plt.tight_layout()
plt.savefig('./res/8.Spatial_Flux_Distribution.png', dpi=300)
#plt.show()

###########################################################################################
################################ LEAKAGE AND ABSORPTION ###################################
###########################################################################################

print(f"{'Material':<15} | {'Leakage (%)':<15} | {'Absorption (%)':<15} | {'Total Loss'}")
print("-" * 70)

for name in results_spectrum.keys():
    # 1. Leakage extraction (Current at the surface)
    total_leak = results_leakage[name]
    rel_leak = (total_leak / 1.0) * 100
    
    # 2. Absorption extraction (Global absorption tally)
    total_abs = results_abs[name]
    rel_abs = (total_abs / 1.0) * 100
    
    # 3. Conservation balance (Should be close to 1.0)
    total_loss = total_leak + total_abs
    
    # Formatted output
    print(f"{name:<15} | {rel_leak:<14.2f}% | {rel_abs:<14.2f}% | {total_loss:.4f}")

    # Small warning if the balance does not close (convergence or tally issue)
    if abs(total_loss - 1.0) > 0.01:
        print("\n[WARNING] Neutron balance deviation > 1%. Check statistics or tally definitions.")
        
    print("-" * 70)