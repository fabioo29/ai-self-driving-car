# TODO
# Add info below the track (increase window dimensions)
# red circles representing inputs (darker = small distance);
# red circles representing output (darker = small distance);
# Generation level, crashed (23/100), best fitness, track number;
# print to the console the weights after the generation ends.

# TODO
# Add PID controller Car to stay on middle of the track
# This car will stop bugged cars (spinning, wrong direction)

# track 1 and track 2 complete: [-0.49, -0.36, -0.99, 1.0, -0.88, 0.13, 0.39, -0.76, 0.67]

import os
import sys
import math
import pygame
import random
import argparse

from math import sin, cos, radians
from argparse import Namespace
from simple_pid import PID

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

    def __init__(self, track_id=None):
        self.track_id = self.get_track_id(track_id)
        self.track = TRACKS[self.track_id]
        self.border = BORDERS[self.track_id]
        self.border_mask = BORDERS_MASK[self.track_id]
        self.width = self.track.get_width()
        self.height = self.track.get_height()
        self.start_x, self.start_y = START_COORDS

    def get_track_id(self, track_id):
        """get the track id"""
        return track_id if track_id else random.randint(0, len(TRACKS) - 1)

    def draw(self):
        """draw the track"""
        WIN.blit(GRASS, (0, 0))
        WIN.blit(self.track, (0, 0))
        WIN.blit(self.border, (0, 0))
        WIN.blit(FINISH_LINE, FL_POS)


class Sensor:
    """class to create a car sensor object"""

    def __init__(self, car, border_mask, degree: int = 0, max_dist=MAX_DIST):
        self.car = car
        self.x = car.x + car.width / 2
        self.y = car.y + car.height / 2
        self.degree = degree
        self.angle = car.angle
        self.border_mask = border_mask
        self.curr_distance = 0
        self.max_dist = max_dist
        self.color = (255, 165, 0)

    def __call__(self):
        return round(self.curr_distance, 2)

    def update(self):
        """update the sensor position"""
        self.x = self.car.x + self.car.width / 2
        self.y = self.car.y + self.car.height / 2
        self.angle = -self.car.angle + self.degree

        # get the sensor measurement
        self.curr_distance = self.get_distance()

    def get_distance(self):
        """get the distance between the sensor and the wall"""
        x1, x2 = self.x, self.x + 1*cos(radians(self.angle))
        y1, y2 = self.y, self.y + 1*sin(radians(self.angle))
        angle = self.angle

        for _ in range(self.max_dist):
            if self.border_mask.get_at((int(x2), int(y2))):
                return get_distance_from_points(x1, y1, x2, y2)
            x2 += cos(radians(angle))  # add one pixel more to x2
            y2 += sin(radians(angle))  # add one pixel more to y2
        else:
            return self.max_dist

    def draw(self):
        pygame.draw.line(
            WIN, self.color,
            (self.x, self.y),
            (self.x + cos(radians(self.angle)) * self.curr_distance,
             self.y + sin(radians(self.angle)) * self.curr_distance), 2
        )


class Car:
    """class to create a car object"""

    def __init__(self, track):
        self.track = track
        self.x = self.track.start_x
        self.y = self.track.start_y
        self.image = CARS[random.randint(0, len(CARS) - 1)]
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.angle = START_ALIGN
        self.max_steering = args.max_steering
        self.velocity = args.max_speed

    def update(self):
        """update car data"""
        self.x += self.velocity * cos(radians(self.angle))
        self.y -= self.velocity * sin(radians(self.angle))

    def draw(self):
        """draw car on screen"""
        blit_rotate(WIN, self.image, (self.x, self.y), self.angle)


class PlayerCar(Car):
    """class to create a USER controlled car object"""

    def __init__(self, track):
        super().__init__(track)
        self.max_speed = self.velocity
        self.acceleration = round(self.max_speed / 10, 2)
        self.rotation_vel = round(self.max_steering / 10, 2)
        self.velocity = 0

    def move(self):
        """move car on screen"""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.velocity += self.acceleration
            self.velocity = min(self.velocity, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.velocity -= self.acceleration
            self.velocity = max(self.velocity, -self.max_speed)
        else:
            if self.velocity > 0:
                self.velocity -= (self.acceleration/2)
                self.velocity = max(self.velocity, 0)
            else:
                self.velocity += (self.acceleration/2)
                self.velocity = min(self.velocity, 0)

        if keys[pygame.K_LEFT] and self.velocity != 0:
            self.angle += self.rotation_vel
        elif keys[pygame.K_RIGHT] and self.velocity != 0:
            self.angle -= self.rotation_vel

        self.update()  # update object coords


class CleanerCar(Car):
    """class to create a PID controlled car object"""

    def __init__(self, track):
        super().__init__(track)
        self.image = CLEANER
        self.Kp = 0.10  # proportional gain
        self.Ki = 0.0002  # integral gain
        self.Kd = 0.00005  # derivative_gain
        self.controller = self.get_controler()
        self.sensors = [
            Sensor(self, self.track.border_mask, degree=-50, max_dist=90),
            Sensor(self, self.track.border_mask, degree=-25, max_dist=90),
            Sensor(self, self.track.border_mask, degree=25, max_dist=90),
            Sensor(self, self.track.border_mask, degree=55, max_dist=90),
            Sensor(self, self.track.border_mask, degree=-90),  # cleaner left
            Sensor(self, self.track.border_mask, degree=90),  # cleaner right
        ]

    def draw_cleaner_line(self):
        """return a line surface merging two objects"""
        obj1, obj2 = self.sensors[-2:]
        cleaner_line = pygame.draw.line(
            WIN, (255, 0, 0),
            (obj1.x + cos(radians(obj1.angle)) * obj1.curr_distance,
             obj1.y + sin(radians(obj1.angle)) * obj1.curr_distance),
            (obj2.x + cos(radians(obj2.angle)) * obj2.curr_distance,
             obj2.y + sin(radians(obj2.angle)) * obj2.curr_distance), 4
        )
        return cleaner_line

    def get_controler(self):
        """get PID controller"""
        my_controller = PID(self.Kp, self.Ki, self.Kd)
        my_controller.setpoint = 0  # setpoint
        my_controller.sample_time = 1/args.fps  # sample time
        return my_controller

    def get_angle(self):
        """get car angle"""
        [sensor.update() for sensor in self.sensors]
        sensors_left_val = sum([self.sensors[0](), self.sensors[1]()])
        sensors_right_val = sum([self.sensors[2](), self.sensors[3]()])

        return self.controller(sensors_left_val - sensors_right_val)

    def move(self):
        self.angle -= self.get_angle()
        self.update()  # update object coords

    def draw(self):
        """draw car on screen"""
        self.cleaner = self.draw_cleaner_line()
        if args.show_cleaner:
            [sensor.draw() for sensor in self.sensors[:-2]]
            blit_rotate(WIN, self.image, (self.x, self.y), self.angle)

    def collide(self, mask, x=0, y=0):
        """check if car has collided with the track border"""
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)

        # if PIDCar crashed stop the simulation
        if poi:
            sys.exit()

    def update(self):
        """update car data"""
        self.x += self.velocity * cos(radians(self.angle))
        self.y -= self.velocity * sin(radians(self.angle))


class PopulationCar(Car):
    """class to create an AGENT controller car object"""

    def __init__(self, track):
        super().__init__(track)

    def move(self):
        pass

    def collide(self, mask, x=0, y=0):
        """check if car has collided with the track border"""
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)

        # if PopulationCar crashed stop it
        if poi:
            self.velocity = 0
            self.image = self.image.convert_alpha()
            self.image = (
                self.image.fill(
                    (128, 128, 128),
                    special_flags=pygame.BLEND_RGB_ADD
                )
            )


def parse_args() -> Namespace:
    """
    Parse args from the CLI or use the args passed in
    :return: Args to be used in the script
    """
    parser = argparse.ArgumentParser(
        description="Genetic algorithm self-driving car simulation")

    parser.add_argument(
        '-t --track',
        dest='track',
        type=int,
        default=None,
        help=f"Racing track to use. Default: None = Random track",
    )

    parser.add_argument(
        '-user-test',
        action="store_true",
        dest='user_test',
        default=True,
        help="Spawn car controlled by user on the track",
    )

    parser.add_argument(
        '--show-cleaner',
        action="store_true",
        dest='show_cleaner',
        default=False,
        help="Show CleanerCar on the track",
    )

    parser.add_argument(
        "-p", "--population",
        type=int,
        default=100,
        help=f"Population for each generation (default: 100 cars)",
    )

    parser.add_argument(
        "-t", "--max-time",
        type=int,
        default=60,
        help=f"Max time for each generation to evolve (default: 60 seconds)",
    )

    parser.add_argument(
        "-fps",
        type=int,
        default=30,
        help=f"Frames per second (default: 30)",
    )

    parser.add_argument(
        "--max-speed",
        type=int,
        dest="max_speed",
        default=5,
        help=f"Max speed of the car (default: 5 px/frame)",
    )

    parser.add_argument(
        "--max-rotation",
        type=int,
        dest="max_steering",
        default=90,
        help=f"Max rotation for the car to aim at (default: 90 degrees)",
    )

    args = parser.parse_args()

    return args


def update_objects(*args):
    """update the car moves"""
    [car.move() for car in args if car]


def draw_window(track, *args):
    """update the window"""
    track.draw()
    [car.draw() for car in args if car]


def check_collisions(*args):
    """check for collisions"""
    [car.collide(car.track.border_mask) for car in args if car]


def main() -> None:
    """
    Main function to run the simulation.
    """

    track = Track(args.track)
    PIDCar = CleanerCar(track)
    myCar = PlayerCar(track) if args.user_test else None
    AgentCar = [PopulationCar(track) if not args.user_test else None]

    clock = pygame.time.Clock()
    while True:
        clock.tick(args.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pos())

        update_objects(myCar, PIDCar, *(AgentCar))
        check_collisions(PIDCar, *(AgentCar))
        draw_window(track, myCar, PIDCar, *(AgentCar))
        pygame.display.update()


if __name__ == '__main__':
    args = parse_args()
    main()
