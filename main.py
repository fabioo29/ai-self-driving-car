import os
import sys
import math
import pygame
import random
import argparse

from math import sin, cos, radians
from argparse import Namespace

from data import WIDTH, HEIGHT, GRASS, FL_POS, CLEANER
from data import START_COORDS, START_ALIGN, MAX_DIST, CARS
from data import TRACKS, BORDERS, BORDERS_MASK, FINISH_LINE

from utils import scale, rotate, blit_rotate, get_distance_from_points

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Genetic algorithm self-driving car simulation')


class Population:
    """class to create a population of car objects"""
    pass


class Track:
    """class to create a track object"""
    pass


class Sensor:
    """class to create a car sensor object"""
    pass


class Car:
    """class to create a car object"""
    pass


class PlayerCar(Car):
    """class to create a USER controlled car object"""
    pass


class CleanerCar(Car):
    """class to create a PID controlled car object"""
    pass


class PopulationCar(Car):
    """class to create an AGENT controller car object"""
    pass


def update_objects(*args):
    """update the car moves"""
    pass


def draw_window(track, *args):
    """update the window"""
    pass


def check_collisions(*args):
    """check for collisions"""
    pass


def main() -> None:
    """
    Main function to run the simulation.
    """

    clock = pygame.time.Clock()
    while True:
        clock.tick(args.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pos())

        update_objects()
        check_collisions()
        draw_window()
        pygame.display.update()


if __name__ == '__main__':
    main()
