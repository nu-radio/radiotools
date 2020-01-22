import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import radiotools.helper as hp
from matplotlib import rc
from collections import Counter
import argparse, textwrap

parser = argparse.ArgumentParser(description=('Selects CoREAS files from a given '
    'list so that they follow an isotropic distribution with a given min/max energy, '
    'spectral index and min/max zenith\n'
    '<<------>> How to use <<------>>\n'
    '1. Create a list of available events using the script make_event_props_list.py\n'
    '2. Run this script and specify the direction and energy distribution you want. '
    'The script will randomly create events following that distribution and select '
    'for each event the most similar CoREAS file by calculating a chi^2-like parameter '
    'chi_2=(delta_direction/sigma_angle)**2 + (delta_log_energy/sigma_log_energy)**2 '
    'and selecting the CoREAS file with the smallest chi_2.\n'
    '3. The script creates plots showing the desired and resulting distributions, '
    'the correlations between parameters and how often CoREAS files were re-used.'
    'Check that the distributions are what you want and uncorrelated and that files'
    'were not re-used too often.'),
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input_file',
    type=str,
    help='List of available files. Run make_event_props_list.py to create it.')
parser.add_argument('n_events',
    type=int,
    help='Number of events to select from the file list')
parser.add_argument('min_energy',
    type=float,
    help='Minimum cosmic ray energy in eV')
parser.add_argument('max_energy',
    type=float,
    help='Maximum cosmic ray energy in eV')
parser.add_argument('spectral_index',
    type=float,
    help='Spectral index of energy distribution')
parser.add_argument('--min_zenith',
    type=float,
    default=0.,
    help='Minimum cosmic ray zenith angle in degrees')
parser.add_argument('--max_zenith',
    type=float,
    default=90.,
    help='Maximum cosmic ray zenith angle in degrees')
parser.add_argument('--sigma_angle',
    type=float,
    default=5.,
    help='Sigma for the cosmic ray direction used when calculating the deviation (in degrees)')
parser.add_argument('--sigma_log_energy',
    type=float,
    default=.2,
    help='Sigma for the log10 of the cosmic ray energy used when calculating the deviation')
args = parser.parse_args()
input_file =args.input_file
n_events = args.n_events
min_energy = args.min_energy
max_energy = args.max_energy
spectral_index = args.spectral_index
min_zenith = args.min_zenith * np.pi / 180.
max_zenith = args.max_zenith * np.pi / 180.
sigma_angle = args.sigma_angle * np.pi / 180.
sigma_log_energy = args.sigma_log_energy

def calculate_chi_square(event_info, zenith, azimuth, energy):
    target_direction = hp.spherical_to_cartesian(zenith, azimuth)
    event_direction = hp.spherical_to_cartesian(float(event_info[2]), float(event_info[3]))
    direction_error = hp.get_angle(target_direction, event_direction)
    energy_error = np.log10(energy) - np.log10(float(event_info[1]))
    return (direction_error/sigma_angle)**2 + (energy_error/sigma_log_energy)**2


zeniths = np.arccos(np.random.uniform(np.cos(max_zenith), np.cos(min_zenith), n_events))
azimuths = np.random.uniform(0, 360, n_events) /180.*np.pi
energies = np.power((np.abs(np.power(max_energy, (spectral_index+1.)) - np.power(min_energy, (spectral_index+1.)))*np.random.uniform(0,1, n_events)), (1./(spectral_index+1.)))
event_props_list = pickle.load(open(input_file, 'br'))

event_filenames = []
found_energies = []
found_zeniths = []
found_azimuths = []

for i in range(len(zeniths)):
    event_chis = []
    for event in event_props_list:
        event_chis.append(calculate_chi_square(event, zeniths[i], azimuths[i], energies[i]))
    best_event = event_props_list[np.argmin(event_chis)]
    event_filenames.append(best_event[0])
    found_energies.append(float(best_event[1]))
    found_zeniths.append(float(best_event[2]))
    found_azimuths.append(float(best_event[3]))
found_energies = np.array(found_energies)
found_zeniths = np.array(found_zeniths)
found_azimuths = np.array(found_azimuths)

fig1 = plt.figure(figsize=(8,8))
ax1_1 = fig1.add_subplot(223, projection = 'polar')
direction_scatter = ax1_1.scatter(found_azimuths, found_zeniths * 180. / np.pi, alpha=.5, s=25, c=np.log10(found_energies), cmap='YlOrRd')
direction_scatter_cbar = plt.colorbar(direction_scatter, ax=ax1_1)
direction_scatter_cbar.set_label(r'$log_{10}(\frac{E}{eV})$')
ax1_1.set_facecolor('silver')
ax1_1.set_ylim([0,90])
ax1_2 = fig1.add_subplot(224)
ax1_2.hist(found_energies, bins = np.logspace(np.log10(min_energy),np.log10(max_energy),int((np.log10(max_energy) - np.log10(min_energy))/.1)+1), edgecolor='k', density=True, label='found events')
ax1_2.hist(energies, bins = np.logspace(np.log10(min_energy),np.log10(max_energy),int((np.log10(max_energy) - np.log10(min_energy))/.1)+1), density=True, histtype='step', label='generated energies')
ax1_2.set_xscale('log')
ax1_2.set_yscale('log')
x_coords = np.logspace(np.log10(min_energy),np.log10(max_energy),int((np.log10(max_energy) - np.log10(min_energy))/.05)+1)
y_coords = 1./(max_energy**(spectral_index+1) - min_energy**(spectral_index+1))*(spectral_index+1)*x_coords**(spectral_index)
ax1_2.plot(x_coords, y_coords, c='k', linestyle='--', label='target distribution')
ax1_2.legend()
ax1_2.set_xlabel('E [eV]')
ax1_2.set_ylabel('$p(E)$')
ax1_3 = fig1.add_subplot(221)
ax1_3.hist(found_zeniths* 180./np.pi, bins = np.arange(0,90,5), edgecolor='k', density=True, label='found events')
ax1_3.hist(zeniths*180./np.pi, bins = np.arange(0,90,5), density=True, histtype='step', label='generated zeniths')
plot_theta = np.arange(min_zenith, max_zenith, 180./np.pi)
ax1_3.plot(plot_theta*180./np.pi, np.pi/180.*np.sin(plot_theta)/(np.cos(min_zenith) - np.cos(max_zenith)), c='k', linestyle='--', label='target distribution')
ax1_3.legend()
ax1_3.set_xlabel(r'$\theta [^{\circ}]$')
ax1_3.set_ylabel(r'$p(\theta)$')
ax1_4 = fig1.add_subplot(222)
ax1_4.hist(found_azimuths*180./np.pi, bins=np.arange(0,360,15), edgecolor='k', density=True, label='found events')
ax1_4.hist(azimuths*180/np.pi, bins=np.arange(0,360,15), density=True, histtype='step', label='generated azimuths')
ax1_4.plot([0,360], [1./360, 1./360], c='k', linestyle='--', label='target distribution')
ax1_4.legend()
ax1_4.set_xlabel(r'$\phi [^{\circ}]$')
ax1_4.set_ylabel(r'$p(\phi)$')

filename_counts = Counter(event_filenames)
fig2 = plt.figure()
ax2_1 = fig2.add_subplot(111)
ax2_1.hist(filename_counts.values(), bins=np.arange(.5, 10.5,1), edgecolor='k')
#ax2_1.set_yscale('log')
ax2_1.set_xlabel('number of times the same file was used')
#ax2_1.set_yticks([1,10, 100])
ax2_1.set_xticks(np.arange(1,10,1))
ax2_1.grid()

corr_bbox = dict(facecolor='white', alpha=.5)
fig3 = plt.figure(figsize=(6, 8))
ax3_1 = fig3.add_subplot(311)
ax3_1.scatter(found_azimuths*180./np.pi, found_zeniths*180./np.pi)
ax3_1.grid()
covar1 = np.cov(found_azimuths, found_zeniths)
ax3_1.text(.03,.9, 'Correlation: {:.2f}'.format(covar1[0][1]/np.sqrt(covar1[0][0]*covar1[1][1])), transform=ax3_1.transAxes, bbox=corr_bbox)
ax3_1.set_xlabel(r'$\phi [^\circ]$')
ax3_1.set_ylabel(r'$\theta [^\circ]$')

ax3_2 = fig3.add_subplot(312)
ax3_2.scatter(found_zeniths*180./np.pi, np.log10(found_energies))
ax3_2.grid()
covar2 = np.cov(found_zeniths, np.log10(found_energies))
ax3_2.text(.03,.9, 'Correlation: {:.2f}'.format(covar2[0][1]/np.sqrt(covar2[0][0]*covar2[1][1])), transform=ax3_2.transAxes, bbox=corr_bbox)
ax3_2.set_xlabel(r'$\theta [^\circ]$')
ax3_2.set_ylabel(r'$log_{10}(E[eV])$')

ax3_3 = fig3.add_subplot(313)
ax3_3.scatter(found_azimuths*180./np.pi, np.log10(found_energies))
ax3_3.grid()
covar3 = np.cov(found_azimuths, np.log10(found_energies))
ax3_3.text(.03,.9, 'Correlation: {:.2f}'.format(covar3[0][1]/np.sqrt(covar3[0][0]*covar3[1][1])), transform=ax3_3.transAxes, bbox=corr_bbox)
ax3_3.set_xlabel(r'$\phi [^\circ]$')
ax3_3.set_ylabel(r'$log_{10}(E[eV])$')

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig1.savefig('event_distributions.png')
fig2.savefig('event_multiplicities.png')
fig3.savefig('correlations.png')
pickle.dump(event_filenames, open('coreas_file_list.p', 'wb'))
