<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">AI self-driving car using a genetic algorithm</h3>

  <p align="center">
    Problem Solving Challenges :: Search in Complex Environments :: Fundamentals of AI 
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#simulation">Simulation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About

|        Population learning (firsts GENs)         |          Cars driving by itselfs (GEN 173)           |
| :----------------------------------------------: | :--------------------------------------------------: |
| ![Product Name Screen Shot](images/learning.gif) | ![Product Name Screen Shot](images/self-driving.gif) |

<div style="text-align: justify">
  
**Motivation**: Academic project for Fundamentals of Artificial Intelligence, M2AI. Use a evolutionary algorithm to learn how to drive a car in a complex environment with the following polynomial function as a controller for the car (w_1*a + w_2*b + w_3*c + w_4*a*b + w_5*a*c + w_6*b*c + w_7*a**2 + w_8*b**2 + w_9*c**2). w_x being the weights the respective agent chromosome genes and 'a,b and c' the distance sensors values.

**Implementation**: Interface was build using Pygame(2D game package for python). The genetic algorithm process consists in Selection, Cross Over, Mutation and bonus step (Random Agents) to avoid population convergence. Later on, was implemented to the game a car with a different behavior from the others (controlled by a PID controller) in order to swipe out the cars with unwanted behaviors (like spinning cars). There's also a car that can be controlled by the user.

**_Tested with_** with four different tracks [Easy, Medium, Medium-Hard and Hard]

**_Built With_** Pygame (Python framework for games)

<!-- TESTING -->

## PID Cleaner car

|              PIDCar cleaning slow cars               |               PIDCar vs Self-driving Agent               |
| :--------------------------------------------------: | :------------------------------------------------------: |
| ![Product Name Screen Shot](images/learning_PID.gif) | ![Product Name Screen Shot](images/self-driving_PID.gif) |

</div>

<!-- SETUP -->

## Setup

     install Requirements with:
        $ pip install -r requirements.txt

<!-- SIMULATION -->

## Simulation

    usage: main.py [-h] [-t TRACK] [--user-test] [--agent-test] [--show-cleaner]
               [--show-simulation] [-p POPULATION] [-fps FPS] [-gif]

    optional arguments:
      -h, --help            show this help message and exit
      -t TRACK, --track TRACK
                            racing track to be used on the simulation. {Default:
                            None(Random track)}
      --user-test           spawn car that can be controlled by user. {Default:
                            False}
      --agent-test          test the simulation with only the best car. {Default:
                            False}
      --show-cleaner        show the car used to draw the Cleaning line. {Default:
                            False}
      --show-simulation     show the population evolving in a simulation.
                            {Default: True}
      -p POPULATION, --population POPULATION
                            how many cars for each generation. {Default: 100}
      -fps FPS              frames per second. {Default: 30}
      -gif, --export-gif    export gif with agent driving in four tracks + bonus.
                            {Default: False}

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

## Contact

Fábio Oliveira - [LinkedIn](https://www.linkedin.com/in/fabioo29/) - fabiodiogo29@gmail.com

Project Link: [https://github.com/fabioo29/ai-self-driving-car](https://github.com/fabioo29/ai-self-driving-car)  
Project built as a Msc. Applied Artificial Intelligence Student.
