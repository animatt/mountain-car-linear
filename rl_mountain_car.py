#!/usr/local/bin/python3

import numpy as np
import RLtoolkit.Tiles.tiles as tiles
import matplotlib.pyplot as plt


#auxilary functions
def argmax(function, S, f_range, weights) :
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
    if position >= lim.distmax :
        game_in_progress = False
        return (game_in_progress, lim.distmax, \
            np.minimum(velocity, lim.velomax) if velocity >= lim.velomin \
            else np.maximum(velocity, lim.velomin))
    else :
        game_in_progress = True
        return (game_in_progress, np.maximum(position, lim.distmin), \
            np.maximum(velocity, lim.velomin) if velocity <= lim.velomax \
            else np.minimum(velocity, lim.velomax))


# initialize agent
tile_width = 0.2125
num_tilings = 8
memsize = 512

epsilon = 0.1
alpha = 0.1
gamma = 0.95


theta = [np.zeros((512, 1)), np.zeros((512, 1)), np.zeros((512, 1))]


def policy(S, weights, epsilon) :
    state = S / tile_width
    rand = np.random.randint(3)
    tl = tiles.tiles(num_tilings, memsize, (state[0], state[1]))
    return np.random.choice((argmax(q, S, range(3), weights), rand), \
        p = (1 - epsilon, epsilon))


def q(S_tiles, A, weights) :
    val = 0
    for index in S_tiles :
        val += weights[A][index]
    return val


converging = True
while converging :

    S = np.random.uniform(low = -0.6, high = -0.4), 0 # position, velocity
    A = policy(S, theta, epsilon)

    S1 = tiles.tiles(num_tilings, memsize, (state[0] / tile_width, state[1] / tile_width))

    theta[A][S1] += alpha * (-1 + gamma * q(S2, A2, theta[A2]) - q(S1, A, theta[A])) # incomplete
    game_in_progress = True
    while game_in_progress :
