import os
import pygame

from utils import scale, get_assets, get_mask

pygame.init()

FONT = os.path.join('assets', 'RifficFree-Bold.ttf')

# track background 'grass.png'
GRASS = get_assets(*('assets', 'grass'))[0]

# racing tracks to train the agent on
TRACKS = get_assets(*('assets', 'track'))

START_COORDS = (120, 36)  # start position coords
START_ALIGN = -180  # car start pointing (0 = left to right)

# track border and respective mask
BORDERS = get_assets(*('assets', 'border'))
BORDERS_MASK = get_mask(BORDERS)

# car model for cleaner
CLEANER = get_assets(*('assets', 'cleaner'))
CLEANER = [scale(car) for car in CLEANER][0]

# game window dimensions
WIDTH = GRASS.get_width()
HEIGHT = GRASS.get_height()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Genetic algorithm self-driving car simulation')

# car models (different colors)
CARS = get_assets(*('assets', 'car'))
CARS = [scale(car) for car in CARS]

MAX_SPEED = 5
MAX_DIST = 90
MAX_STEERING = 90
MAX_FITNESS = 4000

# finish line mask to increase car fitness on cross
FINISH_LINE = get_assets(*('assets', 'finish'))[0]
FL_MASK = get_mask([FINISH_LINE]+[])[0]
FL_POS = (145, 15)

BEST_CAR_CHROMOSOME = [
    [-0.22, -1, 0.58, 0.11, 0.05, 0.8, -0.69, -0.1, 0.64],
    [-0.22, -1, 0.58, 0.11, 0.05, 0.8, -0.69, -0.1, 0.64],
    [-0.22, -1, 0.58, 0.11, 0.05, 0.8, -0.69, -0.1, 0.69],
    [-0.22, -1, 0.58, 0.11, 0.05, 0.8, -0.69, -0.1, 0.68]
]
