import pickle
import h5py
import glob
from NuRadioReco.modules.io.coreas.coreas import get_angles
from NuRadioReco.utilities import units
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Creates a table of event properties for the generate_event.py script to use')
parser.add_argument('input_folders',
    type=str,
    nargs='+',
    help='list of folders in which the event files are located')
parser.add_argument('--min_stations',
    type=int,
    default='0',
    help='all files with fewer than this number of observers in them will be skipped')
parser.add_argument('--output_name',
    type=str,
    default='event_prop_list.p',
    help='name of the output file')
args = parser.parse_args()
folders = args.input_folders
min_number_of_stations = args.min_stations
output_name = args.output_name
if len(output_name.split('.')) > 1:
    if output_name.split('.')[-1] != 'p':
        output_name += '.p'
prop_list = []

for folder in folders:
    filenames = glob.glob(folder + '/*.hdf5')
    files = []
    for filename in filenames:
            if 'highlevel' not in filename:
                files.append(filename)
    print('{} files found in {}'.format(len(files), folder))
    i_used_files = 0
    for file in files:
        corsika = h5py.File(file, 'r')
        if len(corsika['CoREAS']['observers'].values()) >= min_number_of_stations:
            cr_energy = corsika['inputs'].attrs['ERANGE'][0]*units.GeV
            zenith, azimuth, magnetic_field_vector = get_angles(corsika)
            prop_list.append(np.array([file, cr_energy, zenith, azimuth]))
        i_used_files += 1
    print('{} of those files were used'.format(i_used_files))
prop_list = np.array(prop_list)
pickle.dump(prop_list, open(output_name, 'wb'))
