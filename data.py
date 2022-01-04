import pygame
import os

from utils import scale, get_assets, get_mask, get_distance_from_points

# track background 'grass.png'
GRASS = get_assets(*('assets', 'grass'))[0]

# racing tracks to train the agent on
TRACKS = get_assets(*('assets', 'track'))

START_COORDS = (120, 36)  # start position coords
START_ALIGN = -180  # car start pointing (0 = left to right)

# track border and respective mask
BORDERS = get_assets(*('assets', 'border'))
BORDERS_MASK = get_mask(BORDERS)

# car models (different colors)
CARS = get_assets(*('assets', 'car'))
CARS = [scale(car) for car in CARS]

# car model for cleaner
CLEANER = get_assets(*('assets', 'cleaner'))
CLEANER = [scale(car) for car in CLEANER][0]

# game window dimensions
WIDTH = TRACKS[0].get_width()
HEIGHT = TRACKS[0].get_height()

MAX_DIST = 90

# finish line mask to increase car fitness on cross
FINISH_LINE = get_assets(*('assets', 'finish'))[0]
FL_MASK = get_mask([FINISH_LINE]+[])[0]
FL_POS = (145, 15)
