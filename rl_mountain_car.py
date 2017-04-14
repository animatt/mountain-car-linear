#!/usr/local/bin/python3

import numpy as np
import RLtoolkit.Tiles.tiles as tiles
import matplotlib.pyplot as plt


#auxilary functions
def argmax(function, S, f_domain, weights) :
    largest = float('-inf')
    for ind in frange :
        if function(S, ind, weigths) > largest :
            largest = function(S, ind, weigths)
            action = ind
    return action


# initialize env
# grid boundaries : (-1.2, 0.5); velocity boundaries : [-0.7, 0.7];
# starting loc : [-0.6, -0.4)


class lim :
    distmin = -1.2
    distmax = 0.5
    velomin = -0.7
    velomax = 0.7


def bound(position, velocity) :
    vel = np.maximum(lim.velomin, velocity) if velocity <= lim.velomax \
        else lim.velomax
    pos = np.maximum(position + vel, lim.distmin) if position + vel <= lim.distmax \
        else lim.distmax
    game_in_progress = True if pos == lim.distmax else False
    return (game_in_progress, pos, vel)
    # velocity_t+1 = bound(velocity_t + A_t - cos(3 * position_t))
    # position_t+1 = bound(position_t + velocity_t+1)


# initialize agent
tile_width = 0.2125
num_tilings = 8
memsize = 512

epsilon = 0.1
alpha = 0.1
gamma = 0.95


theta = [np.zeros((512, 1)), np.zeros((512, 1)), np.zeros((512, 1))]


def policy(S_tiles, weights, epsilon) :
    rand = np.random.randint(3)
    return np.random.choice((argmax(q, S_tiles, range(3), weights), rand), \
        p = (1 - epsilon, epsilon))


def q(S_tiles, A, weights) :
    val = 0
    for index in S_tiles :
        val += weights[A][index]
    return val


converging = True
while converging :

    pos1, vel1 = np.random.uniform(low = -0.6, high = -0.4), 0 # position, velocity
    S1 = tiles.tiles(num_tilings, memsize, (pos1 / tile_width, vel1 / tile_width))
    A1 = policy(S1, theta, epsilon)

    pos2, vel2 = bound(pos1, vel1 + .001 * (A1 - 1) - .0025 * np.cos(3 * pos1))
    S2 = tiles.tiles(num_tilings, memsize, (pos2 / tile_width, vel2 / tile_width))
    A2 = policy(S2, theta, epsilon)

    theta[A2][S1] += alpha * (-1 + gamma * q(S2, A2, theta) - q(S1, A1, theta))

    game_in_progress = True
    while game_in_progress :
        pos1, vel1 = pos2, vel2
        S1 = S2
        A1 = A2

        game_in_progress, pos2, vel2 \
            = bound(pos1, vel1 + .001 * (A1 - 1) - .0025 * np.cos(3 * pos1))

        S2 = tiles.tiles(num_tilings, memsize, (pos2 / tile_width, vel2 / tile_width))
        A2 = policy(S2, theta, epsilon)

        theta[A1][S1] += alpha * (-1 + gamma * q(S2, A2, theta) - q(S1, A1, theta))
# velocity_t+1 = bound(velocity_t + A_t - cos(3 * position_t))
# position_t+1 = bound(position_t + velocity_t+1)
