# TODO
# Add info below the track (increase window dimensions)
# red circles representing inputs (darker = small distance);
# red circles representing output (darker = small distance);
# Generation level, crashed (23/100), best fitness, track number;
# print to the console the weights after the generation ends.

# TODO:
# - add transparency to car when it hits the wall

import os
import sys
import math
import pygame
import random
import argparse

from math import sin, cos, radians
from argparse import Namespace
from simple_pid import PID

from data import WIDTH, HEIGHT, GRASS
from data import START_COORDS, START_ALIGN
from data import TRACKS, BORDERS, BORDERS_MASK, FL_POS
from data import FL_MASK, CARS, FINISH_LINE, MAX_DIST, CLEANER

from utils import scale, rotate, blit_rotate, get_distance_from_points

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Genetic algorithm self-driving car simulation')


class Population:
    """class to create a population of car objects"""

    def __init__(self, population_size, track_id: int = None):
        self.population_size = population_size
        self.track = Track(track_id)
        self.cars = self.populate_generation()
        self.start_time = pygame.time.get_ticks()
        self.generation = 1
        self.selection = 0
        self.crashed = 0

    def __call__(self) -> None:
        """Evolve into a new generation and keep going."""
        self.cars.sort(key=lambda x: x.fitness, reverse=True)
        self.natural_selection()
        self.crossover()
        self.mutation()
        self.random_agents()
        self.restart()

    def populate_generation(self):
        """Create random cars with different genes"""
        return [PopulationCar(self.track) for _ in range(self.population_size)]

    def draw(self):
        self.track.draw()
        [car.draw() for car in self.cars[1:]]

    def update(self):
        # draw sensors only in the best car alive
        self.cars.sort(key=lambda x: x.fitness, reverse=True)
        self.cars[0].draw(show_sensor=True)

        [car.update() for car in self.cars]

    def natural_selection(self):
        """Process of selecting the fittest individuals from a population."""

        # replace (60~85%) of the current population
        self.selection = random.randint(
            int(len(self.cars)-1 * 0.6),
            int(len(self.cars)-1 * 0.85)
        )
        self.selection = len(self.cars) - self.selection

        ninth_decentil = (self.population_size - int(self.population_size*0.1))
        if (ninth_decentil - self.selection) % 2 != 0:
            self.selection += 1

        self.generation += 1

    def crossover(self):
        """Process of producing offspring from two parents."""
        for i in range(self.selection, self.population_size, 2):
            p1, p2 = i-self.selection, i+1-self.selection
            parent1 = self.cars[p1]
            parent2 = self.cars[p2]
            w1, w2 = parent1.crossover(parent2)
            self.cars[i].chromosome = w1
            self.cars[i + 1].chromosome = w2

    def mutation(self):
        """Process of randomly changing the values of a gene."""
        for car in self.cars[self.selection:]:
            car.mutation()

    def random_agents(self):
        """Process of creating new fresh childs for 10% of population."""
        ninth_decentil = (self.population_size - int(self.population_size*0.1))
        for car in self.cars[ninth_decentil:]:
            car.chromosome = [round(random.uniform(-1, 1), 2)
                              for _ in range(9)]

    def restart(self):
        self.start_time = pygame.time.get_ticks()
        self.track.restart(parser.track)
        self.crashed = 0


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
        if track_id is not None:
            return track_id
        else:
            return random.randint(0, len(TRACKS) - 1)

    def draw(self):
        """draw the track"""
        WIN.blit(GRASS, (0, 0))
        WIN.blit(self.track, (0, 0))
        WIN.blit(self.border, (0, 0))
        WIN.blit(FINISH_LINE, FL_POS)

    def restart(self, track_id):
        """get the track ready for a new generation"""
        self.__init__(track_id)
        self.draw()
        pygame.display.update()


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
        self.max_steering = parser.max_steering
        self.velocity = parser.max_speed

    def update(self):
        """update car data"""
        self.x += self.velocity * cos(radians(self.angle))
        self.y -= self.velocity * sin(radians(self.angle))

    def draw(self):
        """draw car on screen"""
        blit_rotate(WIN, self.image, (self.x, self.y), self.angle)

    def restart(self, track):
        self.__init__(track)


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

    def draw(self):
        """draw car on screen"""
        if parser.user_test:
            blit_rotate(WIN, self.image, (self.x, self.y), self.angle)


class CleanerCar(Car):
    """class to create a PID controlled car object"""

    def __init__(self, track):
        super().__init__(track)
        self.image = CLEANER
        self.wait_time = 15  # frames
        self.x = self.track.start_x + 8
        self.y = self.track.start_y
        self.velocity = 0
        self.Kp = 0.10  # proportional gain
        self.Ki = 0.0002  # integral gain
        self.Kd = 0.00005  # derivative_gain
        self.controller = self.get_controler()
        self.cleaner_line = None
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
        self.cleaner_line = pygame.draw.line(
            WIN, (255, 0, 0),
            (obj1.x + cos(radians(obj1.angle)) * obj1.curr_distance,
             obj1.y + sin(radians(obj1.angle)) * obj1.curr_distance),
            (obj2.x + cos(radians(obj2.angle)) * obj2.curr_distance,
             obj2.y + sin(radians(obj2.angle)) * obj2.curr_distance), 4
        )

    def get_controler(self):
        """get PID controller"""
        my_controller = PID(self.Kp, self.Ki, self.Kd)
        my_controller.setpoint = 0  # setpoint
        my_controller.sample_time = 1/parser.fps  # sample time
        return my_controller

    def get_angle(self):
        """get car angle"""
        [sensor.update() for sensor in self.sensors]
        sensors_left_val = sum([self.sensors[0](), self.sensors[1]()])
        sensors_right_val = sum([self.sensors[2](), self.sensors[3]()])

        return self.controller(sensors_left_val - sensors_right_val)

    def move(self):
        if self.wait_time == 0:
            self.velocity = parser.max_speed
            self.angle -= self.get_angle()
            self.update()  # update object coords
        else:
            self.get_angle()
            self.wait_time -= 1

    def draw(self):
        """draw car on screen"""
        self.draw_cleaner_line()
        if parser.show_cleaner:
            [sensor.draw() for sensor in self.sensors[:-2]]
            blit_rotate(WIN, self.image, (self.x, self.y), self.angle)

    def collide(self, mask, x=0, y=0, finish: bool = False):
        """check if car has collided with the track border"""
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)

        # if PIDCar crashed stop the simulation
        if poi:
            sys.exit()
        return 0


class PopulationCar(Car):
    """class to create an AGENT controller car object"""

    def __init__(self, track):
        super().__init__(track)
        self.fitness = 0
        self.chromosome = [
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene(),
            self.random_gene()
        ]

        self.sensors = [
            Sensor(self, self.track.border_mask),
            Sensor(self, self.track.border_mask, degree=-30),
            Sensor(self, self.track.border_mask, degree=30)
        ]

    def random_gene(self):
        """Random gene for the car (chromosome)."""
        return round(random.uniform(-1, 1), 2)

    def draw(self, show_sensor: bool = False):
        """draw PopulationCar on screen"""
        if parser.show_simulation:
            [sensor.draw() for sensor in self.sensors if show_sensor]
            blit_rotate(WIN, self.image, (self.x, self.y), self.angle)

    def move(self):
        if self.velocity > 0:
            self.angle -= self.get_angle()
            self.update()  # update object coords

    def collide(self, mask, x=0, y=0, finish: bool = False, crash: bool = False):
        """check if car has collided with the track border"""
        car_mask = pygame.mask.from_surface(self.image)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)

        # if PopulationCar crashed stop it
        if poi:
            if finish and poi[0] == 9:
                self.fitness += 50  # add 20 points if car reached the finish line
            elif not finish and self.velocity > 0:
                self.velocity = 0
                return 1
            elif crash and self.velocity > 0:
                self.velocity = 0
                return 1
        return 0

    def crossover(self, parent2):
        """Uniform crossover operation between two parents and return two childrens."""
        A = self.chromosome[:]
        B = parent2.chromosome[:]

        for i in range(len(self.chromosome)):
            if random.random() < 0.5:
                A[i], B[i] = B[i], A[i]

        return A, B

    def mutation(self):
        """Process of randomly changing the values of a gene."""
        randomIndex = random.randint(0, len(self.chromosome) - 1)
        randomValue = random.uniform(-1, 1)
        self.chromosome[randomIndex] += randomValue
        self.chromosome[randomIndex] = max(-1,
                                           self.chromosome[randomIndex])
        self.chromosome[randomIndex] = min(1, self.chromosome[randomIndex])
        self.chromosome[randomIndex] = round(
            self.chromosome[randomIndex], 2)

    def restart(self, track):
        """Get the car ready for a new generation."""
        weights = self.chromosome
        self.__init__(track)
        self.chromosome = weights

    def get_fitness(self):
        """Update the fitness of the car."""
        self.fitness += 5 if self.velocity > 0 else 0

    def get_angle(self):
        """Get the new angle from weights(genes) and sensors distance."""

        # normalize data from sensors
        sensors_data = [
            sensor.curr_distance / math.sqrt(HEIGHT**2 + WIDTH**2) for sensor in self.sensors]

        eq1 = self.chromosome[0]*sensors_data[0]
        eq2 = self.chromosome[1]*sensors_data[1]
        eq3 = self.chromosome[2]*sensors_data[2]
        eq4 = self.chromosome[3]*sensors_data[0]*sensors_data[1]
        eq5 = self.chromosome[4]*sensors_data[0]*sensors_data[2]
        eq5 = self.chromosome[5]*sensors_data[1]*sensors_data[2]
        eq6 = self.chromosome[6]*sensors_data[0]**sensors_data[0]
        eq7 = self.chromosome[7]*sensors_data[1]**sensors_data[1]
        eq8 = self.chromosome[8]*sensors_data[2]**sensors_data[2]
        eq = eq1 + eq2 + eq3 + eq4 + eq5 + eq6 + eq7 + eq8

        # result to be between -x and x degree
        return round(eq / 8 * 1 * self.max_steering, 2)

    def update(self):
        """update car data"""
        self.x += self.velocity * cos(radians(self.angle))
        self.y -= self.velocity * sin(radians(self.angle))
        [sensor.update() for sensor in self.sensors]
        self.get_fitness()


def update_objects(*args):
    """update the car moves"""
    [car.move() for car in args if car]


def draw_window(track, *args):
    """update the window"""
    track.draw()
    [car.draw() for car in args if car]


def check_collisions(*args):
    """check for collisions"""

    crashed = 0
    if len(args) > 1:
        for car in args[1:]:
            if args[0].cleaner_line:
                x, y = args[0].cleaner_line[:2]
                cl_mask = pygame.mask.from_surface(
                    pygame.Surface(
                        (args[0].cleaner_line[2],
                         args[0].cleaner_line[3])
                    )
                )
                crashed += car.collide(cl_mask, x=x, y=y, crash=True)
            car.collide(FL_MASK, *FL_POS, finish=True)

    crashed += sum([car.collide(car.track.border_mask) for car in args if car])
    return crashed


def restart(log, Population, *args):
    """prepare track and cars for a new generation"""
    print(log)  # previous generation log
    Population()  # evolve into new generation
    cars = Population.cars + list(args[1:])
    [car.restart(Population.track) for car in cars if car]


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
        default=1,
        help=f"Racing track to use. Default: None = Random track",
    )

    parser.add_argument(
        '-user-test',
        action="store_true",
        dest='user_test',
        default=False,
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
        "-s", "--show-simulation",
        action="store_true",
        dest='show_simulation',
        default=True,
        help="Show simulation with population",
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
        default=60,
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


def main() -> None:
    """Main function to run the simulation."""

    # make sure user selected a even number of cars
    pop_size = parser.population // 2 * 2

    track = Track(parser.track)
    PIDCar = CleanerCar(track)
    myCar = PlayerCar(track)
    Agents = Population(pop_size, track.track_id)

    pygame.init()
    clock = pygame.time.Clock()
    while True:
        clock.tick(parser.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pos())

        update_objects(myCar, PIDCar, *(Agents.cars))
        Agents.crashed += check_collisions(PIDCar, *(Agents.cars))
        draw_window(track, myCar, PIDCar, *(Agents.cars))
        pygame.display.update()

        # keep printing in the same line refresehing the screen
        log = f'Gen: {Agents.generation} ' + \
            f'Crashed: {Agents.crashed}/{pop_size} ' + \
            f'Track: {Agents.track.track_id+1} ' + \
            f'Best Chromosome: {Agents.cars[0].chromosome} ' + \
            f'Fitness: {Agents.cars[0].fitness} ' + ' '*5
        print(log, end='\r')

        restart(log, Agents, myCar, PIDCar) if Agents.crashed == pop_size else None


if __name__ == '__main__':
    parser = parse_args()
    main()
