### Self-driving car agent in a 2D racing track using a genetic algorithm.

Understanding Population, Chromosome, Genes in this game context.

- Each generation has a population. A population is a group of chromosomes/agents (cars).
- Each agent has a group of genes. Each gene has his weight in the final decision to control the car.
- The car used in the game has only one actuator (steering). This can be a value between -180 and 180 degrees.
- The **steering** value will result from a function with all the genes and inputs from sensors on it.

New generation (population reproduction)

- The cars with the best performance (fitness) will be used to create the new generation of cars.
- Each new population will ge generated through a process of selection, crossover and mutation.
- **Selection**: Choosing how many cars should be deleted/replaced with new ones.
- **Cross Over**: Taking the best 2 or 3 cars(parents) we can swap genes between them to generate new childs.
- **Mutation**: Even after the Cross Over we can add a random value to one of the genes for some childs.
