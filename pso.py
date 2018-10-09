from joblib import Parallel, delayed
from random import random
from numpy import sqrt, array
from numpy.linalg import norm
from copy import deepcopy
from openmdao.api import Problem

import time
import numpy as np
import sys

from math import sin, cos, pi

from iea37_aepcalc import getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML, calcAEP

    # For Python .yaml capability, in the terminal type "pip install pyyaml".
    # An example command line syntax to run this file is "python iea37-aepcalc.py iea37-ex16.yaml"
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])
    # Read necessary values from .yaml files
turb_coords_waste, fname_turb, fname_wr = getTurbLocYAML("iea37-ex64.yaml")
# Get the array wind sampling bins, frequency at each bin, and wind speed
wind_dir, wind_freq, wind_speed = getWindRoseYAML(fname_wr)
# Pull the needed turbine attributes from file
turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
    fname_turb)

# Express speeds in terms of rated speed
wind_speed /= rated_ws
turb_ci /= rated_ws
turb_co /= rated_ws


nt = 64 # Number of turbines
no_AEP = nt * 3.35 * 24.0 * 365.0

def fit(layout):

    turbineX = np.asarray([turb[0] * cos(turb[1]) for turb in layout])
    turbineY = np.asarray([turb[0] * sin(turb[1]) for turb in layout])

    turb_coords = np.recarray(turbineX.shape, coordinate)
    turb_coords.x, turb_coords.y = turbineX / turb_diam, turbineY / turb_diam


# Calculate the AEP from ripped values
    AEP = rated_pwr * calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_ci, turb_co)
    AEP = np.sum(AEP)
    return - AEP # 1.0 - (AEP / no_AEP)

def real_distance(a_p, b_p):
    a = [a_p[0] * cos(a_p[1]), a_p[0] * sin(a_p[1])]
    b = [b_p[0] * cos(b_p[1]), b_p[0] * sin(b_p[1])]
    return sqrt((a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0)


## Inertia weight 0.5+rand/2.0, by: "Inertia weight strategies in particle swarm optimization" by Bansal et al.

def pso(filename_layout, filename_all_lcoe, filename_best_lcoe, n_run):

    np = 20 # Number of particles in swarm
    max_iter = 200
    nD_constraint = 2.0 # Minimum spacing normalised with diameter between turbines (constraint).
    max_r = 3000.0


    vel = array([[[- max_r / 2.0 + random() * max_r, random() * 2.0 * pi] for _ in range(nt)] for _ in range(np)])
    best_local_layout = array([[[0.0, 0.0] for _ in range(nt)] for _ in range(np)])
    best_own_fitness = [float('inf') for _ in range(np)]
    best_global_fitness = float('inf')

    def create_turbine():
        a = [- max_r + 2.0 * random() * max_r, random() * 2.0 * pi]
        return a

    def create():
        return [create_turbine() for _ in range(nt)]

    #  Produce starting positions
    particles = array([create() for _ in range(np)])
    # particles = initial_swarm

    best_layout = create()

    # k = 0.1
    iter = 0
    # history = array([create() for _ in range(np)])
    with Parallel(n_jobs=1) as parallel:
        for iter in range(max_iter):
            start_time2 = time.time()
            fitness = parallel(delayed(fit)(particles[i]) for i in range(np))
            with open(filename_layout, "a", 1) as outf:
                for p in particles:
                    layout = p
                    for t in layout:
                        outf.write("{} {}\n".format(t[0] * cos(t[1]), t[0] * sin(t[1])))
                    outf.write("\n")
            with open(filename_all_lcoe, "a", 1) as outf:
                for p in fitness:
                    outf.write("{}\n".format(p))
            for p in range(np):
                if fitness[p] < best_own_fitness[p]:
                    best_own_fitness[p] = deepcopy(fitness[p])
                    best_local_layout[p] = deepcopy(particles[p])
                if fitness[p] < best_global_fitness:
                    best_global_fitness = deepcopy(fitness[p])
                    best_layout = deepcopy(particles[p])
            # print best_global_fitness
            with open(filename_best_lcoe, "a", 1) as outf:
                    outf.write("{}\n".format(best_global_fitness))
            for p in range(np):
                ## Solving Constrained Nonlinear Optimization Problems with Particle Swarm Optimization by Xiaohui Hu and Russell Eberhart. For 1.49445 learning coefficients.
                vel[p] = 0.729 * vel[p] + 1.49618 * random() * (best_local_layout[p] - particles[p]) + 1.49618 * random() * (best_layout - particles[p])
                # vel[p] = 0.5 * vel[p] + 0.5 * random() * (best_local_layout[p] - particles[p]) + 1.5 * random() * (best_layout - particles[p])

                # Constraining the velocity of the particles to be less than half the maximum_distance in x and y.
                # for n in range(nt):
                #     if vel[p][n][0] > max_r / 2.0:
                #         vel[p][n][0] = max_r / 2.0
                #     if vel[p][n][0] < - max_r / 2.0:
                #         vel[p][n][0] = - max_r / 2.0

                particles[p] = particles[p] + vel[p]

                # Constraining the position of the particles to remain within the circle.
                for n in range(nt):
                    if particles[p][n][0] > max_r:
                        particles[p][n][0] = max_r
                    if particles[p][n][0] < - max_r:
                        particles[p][n][0] = - max_r

            for b in range(np):
                pp = 0
                while pp == 0:
                    pp = 1
                    for i in range(nt):
                        for j in range(nt):
                            if i != j and real_distance(particles[b][i], particles[b][j]) < 65.0 * 2.0 * nD_constraint:
                                particles[b][j] = create_turbine()
                                pp = 0

            # print("Iteration {0:d} - {1}% - {2} s - obj. {3}".format(iter, float(iter) / max_iter * 100.0, time.time() - start_time2, best_global_fitness))


    best = open('final_layout{}_{}.dat'.format(nt, n_run), 'w')
    best_g_fit = open('final_fitness{}_{}.dat'.format(nt, n_run), 'w')
    for i in range(nt):
        best.write('{} {} {} {}\n'.format(best_layout[i][0] * cos(best_layout[i][1]), best_layout[i][0] * sin(best_layout[i][1]), best_layout[i][0], best_layout[i][1]))
    best_g_fit.write('{}\n'.format(best_global_fitness))
    best.close()
    best_g_fit.close()

if __name__ == '__main__':

    def optimise(run):
        n_run = run
        pso("layouts_{}_{}.dat".format(nt, n_run), "costs{}_{}.dat".format(nt, n_run), "best_cost{}_{}.dat".format(nt, n_run), n_run)

    Parallel(n_jobs=8)(delayed(optimise)(i) for i in range(1, 9))

    # for run in range(1, 11):
    #     n_run = run
    #     pso("aep_layout_pso36_{}.dat".format(n_run), "aep_pso_costs36_{}.dat".format(n_run), "aep_best_pso36_{}.dat".format(n_run))
    # xc= [0., 750., 231.7627, -606.7627, -606.7627, 231.7627, 1500., 1299.0381, 750., 0.,            -750., -1299.0381, -1500., -1299.0381, -750., 0., 750., 1299.0381, 2250, 2114.3084,            1723.6, 1125., 390.7084, -390.7084, -1125., -1723.6, -2114.3084, -2250., -2114.3084, -1723.6,            -1125, -390.7084, 390.7084, 1125., 1723.6, 2114.3084, 3000., 2924.7837, 2702.9066, 2345.4944,            1870.4694, 1301.6512, 667.5628, 0., -667.5628, -1301.6512, -1870.4694, -2345.4944, -2702.9066, -2924.7837,            -3000., -2924.7837, -2702.9066, -2345.4944, -1870.4694, -1301.6512, -667.5628, 0., 667.5628, 1301.6512,            1870.4694, 2345.4944, 2702.9066, 2924.7837]     
    # yc= [0., 0., 713.2924, 440.8389, -440.8389, -713.2924, 0., 750., 1299.0381, 1500,            1299.0381, 750., 0., -750., -1299.0381, -1500., -1299.0381, -750., 0., 769.5453,            1446.2721, 1948.5572, 2215.8174, 2215.8174, 1948.5572, 1446.2721, 769.5453, 0., -769.5453, -1446.2721,            -1948.5572, -2215.8174, -2215.8174, -1948.5572, -1446.2721, -769.5453, 0., 667.5628, 1301.6512, 1870.4694,            2345.4944, 2702.9066, 2924.7837, 3000., 2924.7837, 2702.9066, 2345.4944, 1870.4694, 1301.6512, 667.5628,            0., -667.5628, -1301.6512, -1870.4694, -2345.4944, -2702.9066, -2924.7837, -3000., -2924.7837, -2702.9066,-2345.4944, -1870.4694, -1301.6512, -667.5628]

#     xc = []
#     yc = []
#     for t in range(16):
#         xc.append(1300.0 * cos(360.0/16.0 * t))
#         yc.append(1300.0 * sin(360.0/16.0 * t))

#     turbineX = np.asarray(xc)
#     turbineY = np.asarray(yc)

#     turb_coords = np.recarray(turbineX.shape, coordinate)
#     turb_coords.x, turb_coords.y = turbineX / turb_diam, turbineY / turb_diam


# # Calculate the AEP from ripped values
#     AEP = rated_pwr * calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_ci, turb_co)
#     AEP = np.sum(AEP)
#     print(AEP)
    # from math import radians

    # xc = []
    # yc = []

    # for t in range(11):
    #     xc.append(- 1300.0 + 2600.0 / 10.0 * t)
    #     yc.append(0.0)
    # for t in range(5):
    #     xc.append(1300.0 * cos(random() * 2.0 * pi))
    #     yc.append(1300.0 * sin(random() * 2.0 * pi))

 

    # with open("line_layout.dat", "w") as outf:

    #     for a in range(180):
    #         x_a = []
    #         y_a = []
    #         for t in range(16):
    #             x_a.append(xc[t] * cos(radians(a)))
    #             y_a.append(xc[t] * sin(radians(a)))

    #         turbineX = np.asarray(x_a)
    #         turbineY = np.asarray(y_a)

    #         turb_coords = np.recarray(turbineX.shape, coordinate)
    #         turb_coords.x, turb_coords.y = turbineX / turb_diam, turbineY / turb_diam
    #         AEP = rated_pwr * calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_ci, turb_co)
    #         aep = np.sum(AEP)
    #         outf.write("{} {}\n".format(a, aep))
