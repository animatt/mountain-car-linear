#!/usr/local/bin/python3

import numpy as np
import RLtoolkit.Tiles.tiles as tiles
import matplotlib.pyplot as plt


#auxilary functions
def argmax(function, S, f_domain, weights) :
    largest = float('-inf')
    for ind in f_domain :
        if function(S, ind, weights) > largest :
            largest = function(S, ind, weights)
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
    if pos == lim.distmin :
        vel = 0
    game_in_progress = True if pos < lim.distmax else False
    return (game_in_progress, pos, vel)
    # velocity_t+1 = bound(velocity_t + A_t - cos(3 * position_t))
    # position_t+1 = bound(position_t + velocity_t+1)


# initialize agent
tile_width = 0.2125
num_tilings = 8
memsize = 512

epsilon = 0; # 0.08
alpha = 0.0003 #.0005
gamma = 1


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

count = 1
avg_ep_length = 0
converging = True
while converging :

    pos1, vel1 = np.random.uniform(low = -0.6, high = -0.4), 0 # position, velocity
    S1 = tiles.tiles(num_tilings, memsize, (pos1 / tile_width, vel1 / tile_width))
    A1 = policy(S1, theta, epsilon)

    _, pos2, vel2 = bound(pos1, vel1 + .001 * (A1 - 1) - .0025 * np.cos(3 * pos1))
    S2 = tiles.tiles(num_tilings, memsize, (pos2 / tile_width, vel2 / tile_width))
    A2 = policy(S2, theta, epsilon)

    theta[A1][S1] += alpha * (-1 + gamma * q(S2, A2, theta) - q(S1, A1, theta))

    ep_length = 0
    game_in_progress = True
    while game_in_progress :

        pos1, vel1 = pos2, vel2
        S1 = S2
        A1 = A2

        game_in_progress, pos2, vel2 \
            = bound(pos1, vel1 + .001 * (A1 - 1) - .0025 * np.cos(3 * pos1))

        S2 = tiles.tiles(num_tilings, memsize, (pos2 / tile_width, vel2 / tile_width))
        A2 = policy(S2, theta, epsilon)

        theta[A1][S1] += alpha * (-1 + gamma * game_in_progress * q(S2, A2, theta) - q(S1, A1, theta))

        ep_length += 1

    avg_ep_length = (ep_length + count * avg_ep_length) / (count + 1)

    if count % 1000 == 0 :
        epsilon /= 1.001
        alpha /= 1.001
        print('avg_ep_length:', avg_ep_length)
        input('Press enter to continue')
        count = 0; avg_ep_length = 0

    count += 1
# velocity_t+1 = bound(velocity_t + A_t - cos(3 * position_t))
# position_t+1 = bound(position_t + velocity_t+1)
