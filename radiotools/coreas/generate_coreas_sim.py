import os
import stat
import numpy as np
from radiotools import coordinatesystems
from radiotools import helper as hp
from radiotools.atmosphere import models
import sys

from radiotools.atmosphere.cherenkov_radius import get_cherenkov_radius_from_depth

# $_CONDOR_SCRATCH_DIR


def write_sh_geninponly(filename, output_dir, run_dir, corsika_executable,
                         seed, particle_type, event_number, energy, zenith, azimuth,
                         tmp_dir="/tmp", geninp_path="",
                         atm=1, conex=False, obs_level=1564.,
                         flupro="", flufor="gfortran", pre_executionskript=None, parallel=False,
                         B=[19.71, -14.18], thinning=1e-6, ecuts=[1.000E-01, 5.000E-02, 2.500E-04, 2.500E-04]):
    scratchdir = '$TMPDIR/glaser/%i/%i/' % (int(event_number), int(particle_type))
    fout = open(filename, 'w')
    if pre_executionskript is not None:
        fin = open(pre_executionskript, "r")
        fout.write(fin.read())
        fin.close()
    else:
        fout.write('#! /bin/bash\n')
    fout.write('export RUNNR=%06i\n' % int(event_number))
    fout.write('{17}geninp_aera.py -r $RUNNR -s {0} -u {1} -a {2} -z {3} -t {4} -d {5} --atm {6} --conex {7} --obslevel {8} --parallel {9} --Bx {10} --Bz {11} --thinning {12} --ecuts {13} {14} {15} {16} > {5}/RUN$RUNNR.inp\n'.format(seed, energy * 1e-9, 180. * azimuth / np.pi, 180 * zenith / np.pi, particle_type, ".", atm, int(conex), obs_level, int(parallel), B[0], B[1], thinning, ecuts[0], ecuts[1], ecuts[2], ecuts[3], geninp_path))
    fout.close()
    os.chmod(filename, stat.S_IXUSR + stat.S_IWUSR + stat.S_IRUSR)


def write_sh(filename, output_dir, run_dir, corsika_executable,
             seed, particle_type, event_number, energy, zenith, azimuth,
             tmp_dir="/tmp", radiotools_path="",
             hdf5converter="",
             atm=1, conex=False, obs_level=1564.,
             flupro="", flufor="gfortran",
             pre_executionscript_filename=None,
             pre_executionscript=None,
             particles=True,
             parallel=False, parallel_cut=1e-2,
             B=[19.71, -14.18], thinning=1e-6, ecuts=[1.000E-01, 5.000E-02, 2.500E-04, 2.500E-04],
             stepfc=1,
             hdf5_converter_arguments=""):
    scratchdir = '$TMPDIR/glaser/%i/%i/' % (int(event_number), int(particle_type))
    fout = open(filename, 'w')
    if pre_executionscript is not None:
        fout.write(pre_executionscript)
    elif pre_executionscript_filename is not None:
        fin = open(pre_executionscript_filename, "r")
        fout.write(fin.read())
        fin.close()
    else:
        fout.write('#! /bin/bash\n')
    fout.write('export FLUFOR=%s\n' % flufor)
    fout.write('export FLUPRO=%s\n' % flupro)
    fout.write('export TMPDIR=%s\n' % tmp_dir)
    fout.write('export RUNNR=%06i\n' % int(event_number))
    fout.write('cd ' + run_dir + '/\n')
    fout.write('rm -rf {0}$RUNNR\n'.format(scratchdir))
    fout.write('mkdir -p {0}$RUNNR\n'.format(scratchdir))
    executable = os.path.join(radiotools_path, "radiotools", "coreas", "geninp_aera.py")
    fout.write('python {17} -r $RUNNR -s {0} -u {1} -a {2} -z {3} -t {4} -d {5}$RUNNR/ --atm {6} --conex {7} --obslevel {8} --parallel {9} --Bx {10} --Bz {11} --thinning {12} --ecuts {13} {14} {15} {16} --pcut {18} --particles {19} --stepfc {20} > {5}$RUNNR/RUN$RUNNR.inp\n'.format(seed, energy * 1e-9, 180. * azimuth / np.pi, 180 * zenith / np.pi, particle_type, scratchdir, atm, int(conex), obs_level, int(parallel), B[0], B[1], thinning, ecuts[0], ecuts[1], ecuts[2], ecuts[3], executable, parallel_cut, int(particles), stepfc))
    fout.write('cp ' + run_dir + '/SIM$RUNNR.reas {1}$RUNNR/SIM$RUNNR.reas\n'.format(event_number, scratchdir))
    fout.write('cp ' + run_dir + '/SIM$RUNNR.list {1}$RUNNR/SIM$RUNNR.list\n'.format(event_number, scratchdir))
    if (int(particle_type) == 14):
        fout.write('cp ' + run_dir + '/simprot_$RUNNR.reas {1}$RUNNR/SIM$RUNNR.reas\n'.format(event_number, scratchdir))
        fout.write('cp ' + run_dir + '/simprot_$RUNNR.list {1}$RUNNR/SIM$RUNNR.list\n'.format(event_number, scratchdir))
    elif (int(particle_type) == 5626):
        fout.write('cp ' + run_dir + '/simiron_$RUNNR.reas {1}$RUNNR/SIM$RUNNR.reas\n'.format(event_number, scratchdir))
        fout.write('cp ' + run_dir + '/simiron_$RUNNR.list {1}$RUNNR/SIM$RUNNR.list\n'.format(event_number, scratchdir))
    fout.write('cd ' + os.path.dirname(corsika_executable) + '\n')
    fout.write('echo "starting CORSIKA simulation at $(date)"\n')
    if(parallel):
        fout.write('$MPIEXEC $FLAGS_MPI_BATCH ' + os.path.basename(corsika_executable) + ' {0}$RUNNR/RUN$RUNNR.inp > {0}$RUNNR/RUN$RUNNR.out \n'.format(scratchdir))
    else:
        fout.write('./' + os.path.basename(corsika_executable) + ' < {0}$RUNNR/RUN$RUNNR.inp > {0}$RUNNR/RUN$RUNNR.out \n'.format(scratchdir))
    fout.write('echo "CORSIKA simulation finished on $(date)"\n')
    fout.write('rm -rf {0}/$RUNNR.tar.gz\n'.format(output_dir))
    fout.write('mkdir -p {0}\n'.format(output_dir))
    fout.write('cd {0}\n'.format(scratchdir))
    # fout.write('mv RUN$RUNNR.inp {0}steering/RUN$RUNNR.inp\n'.format(dir))
    # fout.write('mv SIM$RUNNR.reas {0}steering/SIM$RUNNR.reas\n'.format(dir))
    # fout.write('mv SIM$RUNNR.list {0}steering/SIM$RUNNR.list\n'.format(dir))
    fout.write('cwd=${PWD##*/}\n')
    fout.write('if [ $cwd != "run" ]\n')
    fout.write('then\n')
#     fout.write('\trm *flout*\n')
#     fout.write('\trm *flerr*\n')
#    fout.write('\ttar --remove-files -cf DAT$RUNNR.tar DAT$RUNNR-*.lst\n')
#     fout.write('\tmkdir {0}\n'.format(os.path.join(output_dir, "../pickle")))
    fout.write('\tmkdir {0}\n'.format(os.path.join(output_dir, "../hdf5")))
    fout.write('\texport PYTHONPATH={0}:$PYTHONPATH\n'.format(radiotools_path))
    executable = os.path.join(radiotools_path, "radiotools", "coreas", "pickle_sim_to_class.py")
    particle_identifier = "xx"
    if (int(particle_type) == 14):
        particle_identifier = "p"
    if (int(particle_type) == 5626):
        particle_identifier = "Fe"
#     fout.write('\tpython {0} -s -d $RUNNR -o {1} --particle-type {2}\n'.format(executable, os.path.join(output_dir, "../pickle"), particle_identifier))
    fout.write('\tpython {} $RUNNR/SIM$RUNNR.reas -o {outputdir} {arguments} \n'.format(hdf5converter, outputdir=os.path.join(output_dir, "../hdf5"), arguments=hdf5_converter_arguments))
    if(parallel):
        # merge particle outputs in case of MPI simulation
        executable = os.path.join(os.path.dirname(corsika_executable), "..", "coast/CorsikaFileIO", "merge_corsika")
        fout.write('\tif [ -f %s ]; then\n' % executable)
        fout.write('\t\techo \"merging particle files\"\n')
        fout.write('\t\t%s -i \"$RUNNR\"/DAT\"$RUNNR\"-?????? -o \"$RUNNR\"/DAT\"$RUNNR\".part\n' % (executable))
        fout.write('\t\tif [ -f \"$RUNNR\"/DAT\"$RUNNR\".part ]; then\n')
        fout.write('\t\t\trm \"$RUNNR\"/DAT\"$RUNNR\"-??????\n')
        fout.write('\t\tfi\n')
        fout.write('\tfi\n')
        fout.write('\tmv \"$RUNNR\"/DAT\"$RUNNR\"-999999999.long \"$RUNNR\"/DAT\"$RUNNR\".long\n')
#     fout.write('\ttar -C $RUNNR --remove-files -czf $RUNNR.tar.gz .\n')
#     fout.write('\tmv $RUNNR.tar.gz {0}/\n'.format(output_dir))
    fout.write('\tmv $RUNNR {0}/\n'.format(output_dir))
    fout.write('\techo "CoREAS output pickleld and zipped and copied to final destination on $(date)"\n')
    # fout.write('\tcd {0}\n'.format(output_dir))
    fout.write('else\n')
    fout.write('\techo "ERROR: still in run directory"\n')
    fout.write('fi\n')
    fout.close()
    os.chmod(filename, stat.S_IXUSR + stat.S_IWUSR + stat.S_IRUSR)


def write_reas(filename, obs_level=1564., run_number=1, event_number=1,
               n=1.000292,
               core_offline=np.array([0, 0, 0]), cs_offline="",
               magnetic_field_declination=0):
    fout = open(filename, 'w')
    fout.write("# CoREAS V1 parameter file\n")
    fout.write("# parameters setting up the spatial observer configuration:\n")
    fout.write("CoreCoordinateNorth = 0                ; in cm\n")
    fout.write("CoreCoordinateWest = 0                ; in cm\n")
    fout.write("CoreCoordinateVertical = %.2f            ; in cm\n" % (obs_level * 100.))
    fout.write("# parameters setting up the temporal observer configuration:\n")
    fout.write("TimeResolution = 2e-10                ; in s\n")
    fout.write("AutomaticTimeBoundaries = 4e-07            ; 0: off, x: automatic boundaries with width x in s\n")
    fout.write("TimeLowerBoundary = -1                ; in s, only if AutomaticTimeBoundaries set to 0\n")
    fout.write("TimeUpperBoundary = 1                ; in s, only if AutomaticTimeBoundaries set to 0\n")
    fout.write("ResolutionReductionScale = 0            ; 0: off, x: decrease time resolution linearly every x cm in radius\n")
    fout.write("# parameters setting up the simulation functionality:\n")
    fout.write("GroundLevelRefractiveIndex = %.8f        ; specify refractive index at 0 m asl\n" % n)
    fout.write("# event information for Offline simulations:\n")
    fout.write("EventNumber = %i\n" % event_number)
    fout.write("RunNumber = %i\n" % run_number)
    fout.write("GPSSecs = 0\n")
    fout.write("GPSNanoSecs = 0\n")
    fout.write("CoreEastingOffline = %.4f                ; in meters\n" % core_offline[0])
    fout.write("CoreNorthingOffline = %.4f                ; in meters\n" % core_offline[1])
    fout.write("CoreVerticalOffline = %.4f                ; in meters\n" % core_offline[2])
    fout.write("OfflineCoordinateSystem = %s                ; in meters\n" % cs_offline)
    fout.write("RotationAngleForMagfieldDeclination = %.6f        ; in degrees\n" % (np.rad2deg(magnetic_field_declination)))
    fout.write("Comment =\n")
    fout.write("CorsikaFilePath = ./\n")
    fout.write("CorsikaParameterFile = RUN%06i.inp\n" % event_number)
    fout.close()


def write_list(filename, station_positions, station_name=None, append=False):
    if not append or not os.path.exists(filename):
        fout = open(filename, 'w')
        fout.close()

    fout = open(filename, 'a')
    for i, position in enumerate(station_positions):
        if station_name is not None:
            name = station_name[i]
        else:
            name = "station_%i" % (i)
        fout.write('AntennaPosition = {0} {1} {2} {3}\n'.format(position[1] * 100., -1 * position[0] * 100., position[2] * 100., name))
    fout.close()


def write_list_star_pattern(filename, zenith, azimuth, 
                            append=False, 
                            obs_level=1400.0, 
                            obs_level_corsika=None, 
                            ground_plane=True,
                            auger_cs=True,
                            inclination=np.deg2rad(-35.7324),
                            r_min=0., r_max=500.,n_rings=20,
                            arm_orientiations=np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]),
                            antenna_rings=None,
                            slicing_method=None, slices=[], 
                            gammacut=None,
                            vxB_plot=False
                            ):
    """
    Parameters
    ----------
    filename :  string
        should have the extension ".list"
        If the file is supposed to be used with the 
        radio_mpi Corsika generator (https://github.com/fedbont94/Horeka/tree/radio_mpi),
        keep the default filename.

    zenith :  float (in radians)
        zenith angle of the incoming signal/air-shower direction (0 deg is pointing vertically upwards)

    azimuth :  float (in radians)
        azimuth angle of the incoming signal/air-shower direction (0 deg is North, 90 deg is West)

    obslevel :  float (!!in m!!)
        Observation level of the detector in the vertical direction

    obs_level_corsika:  float (!!in m!!)
        if the list file should contain more than one observation level, both the
        current observation level as well as the CORSIKA observation level needs
        to be specified as the x, y coordinates are relative to the core position at
        the CORSIKA observation level. The coordinates for all observation levels
        that differ from the CORSIKA observation level need to be shifted along the
        shower axis accordingly.

    ground_plane :  bool     
        True:  for antennas positioned on the ground plane
        False: for antennas positioned in the shower plane, in the air

    auger_cs : bool (default is True)
        True -> you are providing input in Auger coordinates
        False -> you are providing input in Corsika coordinates

    inclination :  float (in radians)
        Inclination of the magnetic field.
        It describes the angle between the Earth's surface and the magnetic field lines.
        It is used to determine the shower plane coordinate system which is heavily dependent on the orientation of the magnetic field.
        The default value is given for the Auger site

    r_min, r_max: float (in m)
        Minimum and maximum radius of antenna rings if no predefined rings are given
    
    n_rings:
        Number of antenna rings if no predefined rings are given
    
    arm_orientations : float(in radians)
        Orientation angles of the arms that make up the antenna rings if no predefined rings are given.
        Number of angles determines amount of antenna arms.
        Default are 8 arms of equal radial distance.

    antenna_rings :  array of antenna ring radii (in m!)
        predefined list of antenna ring radii

    slicing_method, slices: 
        Parameters for viewing the shower at different points in it development

    gamma_cut:

    vxB_plot: bool
        True -> Produce additional list file with antenna position in the shower plane system for visual checks
    """

    # errors that catch when input is in wrong unit
    if obs_level > 10000:
        sys.exit(f"Observation level likely given in cm: {obs_level}. Must be given in meters!")
    
    if np.abs(zenith) > 7:
        sys.exit("Zenith angle likely given in degrees: {zenith}. Must be given in radians!")

    if np.abs(inclination) > 7:
        sys.exit("Magnetic field inclination angle likely given in degrees: {inclination}. Must be given in radians!")

    # make empty .list file if already existent
    if not append or not os.path.exists(filename):
        fout = open(filename, 'w')
        fout.close()

    # open antenna file for writing
    fout = open(filename, 'a')

    if obs_level_corsika is None:
        obs_level_corsika = obs_level

    # compute translation in x and y if corsika observation level is specified, otheriwse shift is zero
    r = np.tan(zenith) * (obs_level - obs_level_corsika)
    deltax = np.cos(azimuth) * r
    deltay = np.sin(azimuth) * r

    # print information about input processing
    print(f"Generating antenna positions at observation level {obs_level} m.")
    print(f"Zenith: {np.rad2deg(zenith)} degrees - in Corsika convention")

    # define angle for Auger rotation 
    # set as 0 degrees for Corsika input
    # Auger coordinates are Corsika coordinates rotated by -90 degrees
    # so: x direction = East, y direction = North
    if auger_cs == True:
          rot_angle = np.deg2rad(270)
          # save corsika azimuth angle for output
          corsika_azimuth = np.round(np.rad2deg(azimuth) - 270, decimals=2)
          # print Corsika input angle for Auger input
          print(f"Azimuth: {corsika_azimuth} degrees - in Corsika convention")


    elif auger_cs == False:
          rot_angle = 0
          # save corsika azimuth angle for output
          corsika_azimuth = np.round(np.rad2deg(azimuth) - 180, decimals=2)
          # print Corsika input angle
          print(f"azimuth: {corsika_azimuth} degrees - in Corsika convention")

    else:  # dealing with wrong input choices:
        sys.exit("Invalid input. Possible options for Auger_CS are 'True' or 'False'. \n Quitting...")


    print("These are the angles that should be in the Corsika input file!!!")

    # rotation matrix for transformation between Auger and Corsika coordinate system
    # rotation matrix for rotation around z-axis
    # if Auger_CS=False, this is an identity matrix
    rotation_z_axis = np.array([[np.cos(rot_angle),  (-1) * np.sin(rot_angle), 0], \
                      [np.sin(rot_angle), np.cos(rot_angle), 0], \
                      [0, 0, 1]])
    
    # inverse rotation matrix for magnetic field vector
    inverse_rotation = np.linalg.inv(rotation_z_axis)
 
    # compute the B field in Corsika system (x direction = North, y direction = West) from inclination of geomagnetic field given in input
    B_field = np.array([np.cos(inclination), 0, -np.sin(inclination)])
    print("Magnetic field vector: ", B_field)
    print("Magnetic field inclination", np.rad2deg(inclination))

    # rotate magnetic field vector vertical axis in opposite direction of station coordinates
    # depends on Auger_CS
    B_field = np.dot(inverse_rotation, B_field)

    # define shower plane coordinate system from given geometry
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=B_field)

    # string for the end of the antenna names
    observation_plane_string = "gp" # short for groundplane
    if not ground_plane:
        # instead use shower plane shorthand
        observation_plane_string = "sp"

    # check whether antenna ring radii are provided by input
    if antenna_rings is None:
        antenna_rings = np.linspace(r_min, r_max, n_rings + 1)

    # if provided, add an additional antenna in the middle
    else:
        n_rings = len(antenna_rings)
        antenna_rings = np.append(0, antenna_rings)

    # array to save all station positions in
    station_positions_groundsystem = []

    # loop to put define antennas at the specified positions
    for i in np.arange(1, n_rings + 1): # loop over number of antenna rings
        for j in np.arange(len(arm_orientiations)): # loop over number of arms
             # place antennas along arm in shower plane coordinates
            station_position = antenna_rings[i] * hp.spherical_to_cartesian(np.pi * 0.5, arm_orientiations[j])
            # set antenna name
            name = "pos_%i_%i_%.0f_%s" % (antenna_rings[i], np.rad2deg(arm_orientiations[j]), obs_level, observation_plane_string)

            # for ground plane antenna array
            if ground_plane:
                # transform station positions to ground plane coordinates and set all the z coordinates to 0
                pos_2d = cs.transform_from_vxB_vxvxB_2D(station_position)  # position if height in observer plane should be zero
                # add xy shift if applicable
                pos_2d[0] += deltax
                pos_2d[1] += deltay

                # write transformed coordinates into kartesian vector and 
                # set z coordinate to observation level
                # and finally convert to cm (Corsika's favourite unit)
                antennas = np.array([100 * pos_2d[0], 100 * pos_2d[1], 100 * obs_level])

                # write all station positions into list for plot in vxB coordinates
                station_positions_groundsystem.append(antennas)

                # apply rotation matrix to stations
                # Corsika input will stay the same, Auger input will be rotated by -90 degrees
                antennas = np.dot(rotation_z_axis, antennas)

            else:
                # transform station positions to ground plane coordinates
                pos = cs.transform_from_vxB_vxvxB(station_position)
                # add xy shift if applicable
                pos[0] += deltax
                pos[1] += deltay

                # write transformed coordinates into kartesian vector and 
                # add observation level to z coordinate
                # and finally convert to cm (Corsika's favourite unit)
                antennas = np.array([100 * pos[0], 100 * pos[1], 100 * (pos[2] + obs_level)])

                # write all station positions into list for plot in vxB coordinates
                station_positions_groundsystem.append(antennas)

                # apply rotation matrix to stations
                # Corsika input will stay the same, Auger input will be rotated by -90 degrees
                antennas = np.dot(rotation_z_axis, antennas)

            if(slicing_method is None):
                # default: with no slicing or gammacut
                if gammacut is None:
                    # save the generated starshapes to the antenna list file
                    # positions in cm
                    fout.write('AntennaPosition = {0} {1} {2} {3}\n'.format(antennas[0], antennas[1], antennas[2], name))
                else:
                    for iG, gcut in enumerate(gammacut):
                        name = "pos_%i_%i_gamma%i" % (antenna_rings[i], np.rad2deg(arm_orientiations[j]), iG)
                        # save the generated starshapes to the antenna list file
                        # positions in cm
                        fout.write('AntennaPosition = {0} {1} {2} {3} gamma {4} {5}\n'.format(antennas[0], antennas[1], antennas[2], name, gcut[0], gcut[1]))
            else:
                if(len(slices) <= 1):
                    print("ERROR: at least one slice must be specified")
                    raise
                if(not (slicing_method == "distance" or slicing_method == "slantdepth")):
                    print("ERROR: slicing method must be either distance or slantdepth")
                    raise
                slices = np.array(slices)
                if(slicing_method == "distance"):
                    slices *= 100
                for iSlice in range(len(slices) - 1):
                    name = "pos_%i_%i_slice%i" % (antenna_rings[i], np.rad2deg(arm_orientiations[j]), iSlice)
                    if gammacut is None:
                        # save the generated starshapes to the antenna list file
                        # positions in cm
                        fout.write('AntennaPosition = {0} {1} {2} {3} {4} {5} {6}\n'.format(antennas[0], antennas[1], antennas[2], name, slicing_method, slices[iSlice] * 100., slices[iSlice + 1] * 100.))
                    else:
                        for iG, gcut in enumerate(gammacut):
                            name_gamma = "%s_gamma%i" % (name, iG)
                            # save the generated starshapes to the antenna list file
                            # positions in cm
                            fout.write('AntennaPosition = {0} {1} {2} {3} {4} {5} {6} gamma {7} {8}\n'.format(antennas[0], antennas[1], antennas[2], name_gamma, slicing_method, slices[iSlice] * 100., slices[iSlice + 1] * 100., gcut[0], gcut[1]))
    
    print("Saved antenna positions (in groundplane coordinates) to file: ", filename)

    fout.close()

    # in case you want to plot the antennas in the shower plane coordinate system
    # mainly for visual checking whether the starshape is ok
    if vxB_plot:
        # open the shower.list file to save the generated starshapes to
        with open("shower_plane.list", "w") as file:
                
            # transform the station positions to vxB system for plot
            shower_plane_system = cs.transform_to_vxB_vxvxB(np.array(station_positions_groundsystem))
         
            for i in range(len(shower_plane_system)):
                # save the generated starshapes to the antenna.list file
                # positions in cm
                file.write(f"AntennaPosition = {shower_plane_system[i, 0]} {shower_plane_system[i, 1]} {shower_plane_system[i, 2]} {name}\n")
            
            print("Saved antenna positions (in vxB_vxvxB coordinates) to file: ", "shower_plane.list")


    # return corsika azimuth angle to for automatically generating corsika input files with the right values
    return corsika_azimuth


def get_starshaped_pattern_radii(zenith, obs_level, n0=1.000292, at=None, atm_model=None):
    """
    function to generate starshape antenna pattern with certain features:

    Dense core 

    Dense rings up to just outside the cherenkov ring

    Sparse up to multiple times of the cherenkov radius
    """

    # This is just validated for has shower
    # is not even sopisticated

    # obs_level has to be given in m
    # zenith must be given in radians

    # errors that catch when input is in wrong unit
    if obs_level > 10000:
            sys.exit(f"Observation level likely given in cm: {obs_level}. Must be given in meters!")
    
    if zenith > 7:
            sys.exit("Zenith angle likely given in degrees. Must be given in radians!")

    if at is None:
        if atm_model is None:
            sys.exit("No proper arguments for get_starshaped_pattern_radii")

        at = models.Atmosphere(atm_model)

    # calculate cherenkov radius from zenith angle, depth of maximum, observation level, and atmosphere model
    # uses 750 g/cm² as an approximation
    # THIS IS ONLY (APPROX.) VALID FOR PROTONS AT ENERGIES: 10e16 - 10e19 eV
    cherenkov_radius = get_cherenkov_radius_from_depth(zenith=zenith, depth=750, obs_level=obs_level, n0=n0, model=atm_model) # returns in m

    # max radius of antenna rings, good balance for getting most of the emission and not simulating too far out to be useful
    rmax = cherenkov_radius * 6

    # safe distance where you get most of the cherenkov radii with the dense array
    r_cherenkov_upper_limit = (cherenkov_radius * 1.23 + 80)

    # create list of antenna rings with denser rings within cherenkov radius and a little beyond
    antenna_rings = np.append(0.005 * rmax, np.append(
                   np.linspace(0.01 * rmax, r_cherenkov_upper_limit, 14, endpoint=False),
                   np.linspace(r_cherenkov_upper_limit, rmax, 15)))
    
    # all lengths here are given in m for the input of the starshape generator function
    # conversion to cm for the corsika output happens in that function!

    return antenna_rings


# def write_list_multiple_heights(filename, zen, az, obs_level=[1564., 0.],
#                                 inc=np.deg2rad(-35.7324), zero_height=True, r_min=0., r_max=500.,
#                                 slicing_method="", slices=[], n_rings=20,
#                                 arm_orientiations=np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]),
#                                 atm_model=1):
#     """
#     inc is the inclination of the magnetic field from CoREAS repo (at AERA site)
#     """
#     fout = open(filename, 'w')
#     B = np.array([0, np.cos(inc), -np.sin(inc)])
#     cs = CSTransformation.CSTransformation(zen, az, magnetic_field=B)
#
#     def get_rmax(X):
#         return 0.37 * X + 2e-5 * X ** 2  # LDF_falloff2.pdf
#
#     for h in obs_level:
#
#         # compute translation in x and y
#         r = np.tan(zen) * (h - 0)
#         deltax = np.cos(az) * r
#         deltay = np.sin(az) * r
#
#         from atmosphere import models as atm
#         d = h / np.cos(zen)
#         h2 = atm.get_height_above_ground(d, zen, observation_level=0)
#         Xst = atm.get_atmosphere2(zen, h_low=h2, model=atm_model)
#         # Xst = atm.get_atmosphere(h) / np.cos(zen)
#         rmax = get_rmax(Xst)
#         rs = np.append(0.005 * rmax, np.append(np.linspace(0.01 * rmax, rmax * 0.15, 12),
#                        np.linspace(rmax * 0.20, rmax, 17)))
#
#         for i, r in enumerate(rs):
#             for j in np.arange(len(arm_orientiations)):
#                 station_position = rs[i] * hp.SphericalToCartesian(np.pi * 0.5, arm_orientiations[j])
#                 pos = cs.transform_from_vxB_vxvxB(station_position)
#                 pos_2d = cs.transform_from_vxB_vxvxB_2D(station_position)  # position if height in observer plane should be zero
#
#                 pos[0] += deltax
#                 pos[1] += deltay
#                 pos_2d[0] += deltax
#                 pos_2d[1] += deltay
#
#                 name = "pos_%i_%i_%.0f_%.0f" % (rs[i], np.rad2deg(arm_orientiations[j]), Xst, h)
#                 x, y, z = 100 * pos[1], -100 * pos[0], 100 * (pos[2] + h)
#                 if(zero_height):
#                     x, y, z = 100 * pos_2d[1], -100 * pos_2d[0], 100 * h
#                 if(slicing_method == ""):
#                     fout.write('AntennaPosition = {0} {1} {2} {3}\n'.format(x, y, z, name))
#                 else:
#                     if(len(slices) <= 1):
#                         print("ERROR: at least one slice must be specified")
#                         raise
#                     if(not (slicing_method == "distance" or slicing_method == "slantdepth")):
#                         print("ERROR: slicing method must be either distance or slantdepth")
#                         raise
#                     slices = np.array(slices)
#                     if(slicing_method == "distance"):
#                         slices *= 100
#                     for iSlice in range(len(slices) - 1):
#                     for iSlice in range(len(slices) - 1):
#                         fout.write('AntennaPosition = {0} {1} {2} {3} {4} {5} {6}\n'.format(x, y, z, name, slicing_method, slices[iSlice] * 100., slices[iSlice + 1] * 100.))
#     fout.close()


def get_radius(zenith, dxmax, rel=0.01):
    Amax = LDF2D(0, 0, zenith, dxmax)
    for x in np.arange(10, 2000, 10):
        if(LDF2D(x, 0, zenith, dxmax) < (rel * Amax)):
            return x
    return 100


def LDF2D(x, y, zenith, dxmax, cx=0, cy=0):
    c0 = 0.41
    c3 = 16.25
    c4 = 0.0079
    if zenith < np.deg2rad(10):
        c1 = 8.
        c2 = -21.2
    elif zenith < np.deg2rad(20):
        c1 = 10.
        c2 = -23.1
    elif zenith < np.deg2rad(30):
        c1 = 12.
        c2 = -25.5
    elif zenith < np.deg2rad(40):
        c1 = 20.
        c2 = -32.
    elif zenith < np.deg2rad(50):
        c1 = 25.1
        c2 = -34.5
    elif zenith < np.deg2rad(55):
        c1 = 27.3
        c2 = -9.8
        c0 = 0.46
    elif zenith >= np.deg2rad(55):
        c1 = 27.3
        c2 = -9.8
        c0 = 0.71
    # dxmax = hp.get_distance_xmax(xmax, zenith, observation_level)
    s1 = 37.7 - 0.006 * dxmax + 5.5e-4 * dxmax ** 2 - 3.7e-7 * dxmax ** 3
    s2 = 26.9 - 0.041 * dxmax + 2e-4 * dxmax ** 2
    A = 1.  # only relative scaling is important
    p1 = A * np.exp(-((x - cx + c1) ** 2 + (y - cy) ** 2) / s1 ** 2)
    p2 = A * c0 * np.exp(-((x - cx + c2) ** 2 + (y - cy) ** 2) / (s2) ** 2)
    return p1 - p2


if __name__ == "__main__":
    write_list("/tmp/test_write_list.list", np.deg2rad(40), 63. / 12. * np.pi, zero_height=False)
