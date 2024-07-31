#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from optparse import OptionParser

# Parse commandline options
parser = OptionParser()
parser.add_option("-r", "--runnumber", default="0", help="number of run")
parser.add_option("-s", "--seed", default="1", help="seed 1")
parser.add_option("-u", "--energy", default="1e8", help="cr energy (GeV)")
parser.add_option("-t", "--type", default="14", help="particle type ")
parser.add_option("-a", "--azimuth", default="0", help="azimuth (degrees; AUGER definition)")
parser.add_option("-z", "--zenith", default="0", help="zenith (degrees; AUGER definition) ")
parser.add_option("-d", "--dir", default="/tmp", help="output dir")
parser.add_option("--atm", default="1", help="atmosphere")
parser.add_option("--conex", default=False, help="use conex")
parser.add_option("--particles", default=True, help="write particle ouput")
parser.add_option("--obslevel", default=1564, help="observation level in m")
parser.add_option("-p", "--parallel", default=False, help="use parallel version of CORSIKA")
parser.add_option("--Bx", default="19.71", help="magnetic field in x direction in muT")
parser.add_option("--Bz", default="-14.18", help="magnetic field in z direction in muT")
parser.add_option("--thinning", default="1e-6", help="thinning level")
parser.add_option("--ecuts", type="float", nargs=4, default=(1.000E-01, 5.000E-02, 2.500E-04, 2.500E-04), help="energy cuts (hardonns, muons, e+/e-, gamma")
parser.add_option("--pcut", type="float", nargs=1, default=1e-2, help="maximal energy for parallel showers as a fraction of cosmic-ray energy.")
parser.add_option("--stepfc", default=1., type="float", help="stepfc corsika parameter. Factor by which the multiple scattering length for electrons and positrons in EGS4 simulations is elongated relative to the value given in [17]")

(options, args) = parser.parse_args()
options.conex = bool(int(options.conex))
options.parallel = bool(int(options.parallel))
options.particles = bool(int(options.particles))
thinning = float(options.thinning)

print("RUNNR", int(options.runnumber), "                               run number")
print("EVTNR   1                              number of first shower event")
if options.parallel:
    print("PARALLEL      %f     %f   1   F" % (1000., options.pcut * float(options.energy)))
print("SEED", int(options.seed), " 0 0")
print("SEED", int(options.seed) + 1, " 0 0")
print("SEED", int(options.seed) + 2, " 0 0")
if options.parallel:
    print("SEED", int(options.seed) + 3, " 0 0")
    print("SEED", int(options.seed) + 4, " 0 0")
    print("SEED", int(options.seed) + 5, " 0 0")
print("NSHOW   1                              number of showers to generate")
print("PRMPAR", options.type, "                             primary particle")
print("ERANGE", float(options.energy), float(options.energy), "                    range of energy")
print("THETAP", float(options.zenith), float(options.zenith), "                     range of zenith angle (degree)")
print("PHIP", -270 + float(options.azimuth), -270 + float(options.azimuth), "                      range of azimuth angle (degree)")
# print "ECUTS   3.000E-01 3.000E-01 4.010E-04 4.010E-04"
# print "ECUTS   1.000E-01       5.000E-02       2.500E-04       2.500E-04"  # ecuts from rdobserver
print("ECUTS   %.4g    %.4g    %.4g     %.4g" % options.ecuts)
print("ELMFLG  T   T                          em. interaction flags (NKG,EGS)")
print("THIN    ", thinning, "  ", thinning * float(options.energy), " 0.000E+00")
print("THINH   1.000E+02 1.000E+02")
print("STEPFC ", options.stepfc)
print("OBSLEV  %.0f                       observation level (in cm)" % (float(options.obslevel) * 100))
print("ECTMAP  1.E5                           cut on gamma factor for printout")
print("MUADDI  T                              additional info for muons")
print("MUMULT  T                              muon multiple scattering angle")
print("MAXPRT  1                              max. number of printed events")
# print "MAGNET  18.4   -14.0                   magnetic field AERA"
# print "MAGNET  19.71  -14.18                    magnetic field AERA CoREAS Repo"
print("MAGNET  %s   %s        Bx and Bz component in micro Tesla            " % (options.Bx, options.Bz))
if options.conex:
    print("PAROUT  F F")
    print("CASCADE T T T")
else:
    if options.particles:
        print("PAROUT  T F")
    else:
        print("PAROUT  F F")
#    print "CASCADE F F F"
print("LONGI   T  10.  T  T                    longit.distr. & step size & fit & out")
print("RADNKG  5.e5                           outer radius for NKG lat.dens.distr.")
print("ATMOD %s" % options.atm)
print("DIRECT", options.dir, "                             output directory")
print("DATBAS  F                              write .dbase file")
print("USER    glaser                           user")
print("EXIT                                   terminates input")
