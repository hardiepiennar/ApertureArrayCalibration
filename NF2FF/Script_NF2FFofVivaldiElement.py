import numpy as np

"""General parameters"""
f = 1e9
lambd = 3e8/f
aut_aperture = 0.42
max_farfield_angle = 15*np.pi/180
probe_aperture = 0.15
probe_aut_seperation = 10

"""Calculate the start and the stop of the radiating nearfield"""
nearfield_start, nearfield_stop = NF2FF.calc_nearfield_bounds(f, aut_aperture)
print("Start of Nearfield: "+str(nearfield_start))
print("Start of Farfield: "+str(nearfield_stop))

"""Calculate recommended parameters for farfield scan"""
probe_gain, scan_width, no_samples_per_ray, sample_delta = NF2FF.calc_recommended_sampling_param(aut_aperture,
                                                                                                 max_farfield_angle,
                                                                                                 probe_aperture,
                                                                                                 probe_aut_seperation,
                                                                                                 lambd)
print("Probe Gain: "+str(10*np.log10(probe_gain)))
print("Scan Width: "+str(scan_width))
print("Samples per Ray: "+str(no_samples_per_ray))
print("Sample delta: "+str(sample_delta))