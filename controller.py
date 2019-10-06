import ea as ea;
import robotic_system as rs; #the nn and the robotic model is here
import numpy as np;
import math as math;
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time


POPULATION_SIZE = 80
SELECTION_COUNT = 50
MUTATION_PROBABILITY = 0.2
SPLIT_POINT = 26
PLOT_COUNTER = 1
PATH_LIST = []
APPROX_TOTAL_BLOCKS = 500


def calculateFitness(velocity_vector, max_strength, cleaned_blocks, previous_cleaned_blocks):
    """ Calculates fitness based on the velocity vector and the area that robot has cleaned """

    translational_motion_factor = abs(1 - math.sqrt( abs( velocity_vector[0] - velocity_vector[1] ) ) /math.sqrt(4) )
    # makes sure that root goes into translation motion

    speed_maximizing_factor = (abs(velocity_vector[0]) + abs(velocity_vector[1])) / 4
    # makes sure that robot has maximum speed


    #collision_avoidance_factor = 1. - max_strength/10.0
    #print('collision_avoidance_factor is ' + str(collision_avoidance_factor) )

    #cleaning_improvement_factor = cleaned_blocks - previous_cleaned_blocks
    #print('cleaning_improvement_factor is: ' + str(cleaning_improvement_factor))
    #print('previous cleaned_area is ' + str(previous_cleaned_blocks))
    #print('cleaned_area is ' + str(cleaned_blocks))

    return 2 * translational_motion_factor + 2 * speed_maximizing_factor +  8 * cleaned_blocks/APPROX_TOTAL_BLOCKS

def plot_figure(x, y, theta):
    """ Plots the points on graphs """
    global PLOT_COUNTER
    global PATH_LIST

    ax = fig.add_subplot(111, aspect='equal')
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, color='green'))
    ax.text(1, 1, 'Space where RobotX can move', color='white')

    ax.add_patch(patches.Rectangle((25, 25), 50, 50, color='brown'))
    ax.text(26, 26, 'Space where RobotX can\'t move', color='white', fontsize=8)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    #plt.scatter(x, y)
    #print(x, y)
    #PATH_LIST.append(tuple((x, y)))
    #print(PATH_LIST)
    #for path in PATH_LIST:
    #    ax.add_patch(patches.Circle((path[0], path[1]), 2, color='yellow'))

    ax.add_patch(patches.Circle((robo_x.x, robo_x.y), 2, color='yellow'))

    ax.add_patch(patches.Arrow(x, y,
                               5 * robo_x.translational_velocity * math.cos(theta ),
                               5 * robo_x.translational_velocity * math.sin(theta ),
                               width=5,color='white'))
    plt.xlabel('X is in meters. X Position: {:0.2f}'.format(robo_x.x))
    plt.ylabel('Y in meters. Y Position: {:0.2f}'.format(robo_x.y))
    #plt.title('Translational velocity: ' + str(round(robo_x.translational_velocity, 2)) + ' meters/second')
    plt.title("Translational velocity: {:0.2f}".format(robo_x.translational_velocity) + ' meters/second')
    plt.pause(0.001)




if __name__ == "__main__":

    # Initialization of Robot

    robo_x = rs.Robot()
    ev_alg = ea.Generation(POPULATION_SIZE,SELECTION_COUNT)
    nn = rs.Neural_Net()

    # infrared sensors output is updated within the class variables
    robo_x.sense_environment(robo_x.x, robo_x.y)

    fig = plt.figure()
    plot_figure(robo_x.x, robo_x.y, robo_x.theta)

    np_max_fitness = np.zeros(300)
    np_avg_fitness = np.zeros(300)
    plt.pause(0.0007)

    previous_cleaned_blocks = 0

    # returns the number of blocks in grid robot has cleaned till now
    cleaned_blocks = robo_x.clean_dirt(robo_x.x, robo_x.y)

# Running for 300 generations
    for i in range(300):

        sum_fitness = 0
        max_fitness = - math.inf

        for gene in ev_alg.genes:
            # For every gene in the population

            weightVector = gene.weightVector
            # reshaping the weights into matrices
            weight_matrix ,feedback_matrix = np.split(weightVector,[SPLIT_POINT])
            weight_matrix = np.reshape(weight_matrix,(nn.input_nodes + 1,nn.output_nodes))
            feedback_matrix = np.reshape(feedback_matrix,(nn.output_nodes,nn.output_nodes))

            # Inputting the weights to Neural Network and calculatiing the velocities
            nn.feedback_network(robo_x.np_sensor_output,weight_matrix,feedback_matrix) #have to change the sensor

            # Transforming NN output to velocities
            velocity_vector = nn.transform_velocity()

            max_strength = np.max(robo_x.np_sensor_output)

            # Calculating Fitness
            gene.fitness = calculateFitness(velocity_vector, max_strength, cleaned_blocks, previous_cleaned_blocks)  #fitness value is calculated here

            # Calculating max fitness for the generation
            if gene.fitness > max_fitness:
                max_fitness = gene.fitness

            # calculating sum of the fitness values to compute average fitness
            sum_fitness = sum_fitness + gene.fitness

            # calculating new position
            robo_x.motion_model(velocity_vector[0],velocity_vector[1])
            plot_figure(robo_x.x, robo_x.y, robo_x.theta)


            previous_cleaned_blocks = cleaned_blocks
            cleaned_blocks = robo_x.clean_dirt(robo_x.x, robo_x.y)

            # Checking for collisions
            robo_x.collision_check()

            # updating Sensor Output
            robo_x.sense_environment(robo_x.x, robo_x.y)
            plt.clf()

        np_avg_fitness[i] = sum_fitness/POPULATION_SIZE
        np_max_fitness[i] = max_fitness


        # EA Selection
        ev_alg.selection()

        # EA Reproduction
        ev_alg.reproduction()

        # EA Mutation
        ev_alg.mutation(MUTATION_PROBABILITY)

    plt.figure()
    plt.plot(np_avg_fitness)
    plt.plot(np_max_fitness)
    plt.show()