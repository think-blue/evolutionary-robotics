import math
import numpy as np
import random

NP_GRID = np.zeros((26,26))
TIME_STEP = 1               # Time step in seconds after which robots updates it's position

class Environment:

    @staticmethod
    def check_location(x, y):
        """ Maps the two dimensional coordinate system to the
         environment in which robot is allowed to move.
        It return 0 for illegal locations and 1 otherwise.
        The environment is 100 X 100 Meters area park"""

        if x <= 0 + 1 or x >= 100 - 1 or y <= 0 + 1 or y >= 100 - 1:
            return 0

        if x >= 25 - 1 and x <= 75 + 1:
            if y >= 25 - 1 and y <= 75 + 1:
                return 0
        return 1

class Robot:

    def __init__(self):
        self.number_of_wheels = 2                           # shape of robot is circle with unit radius
        self.distance_between_wheels = 4                    # Distance between the two wheels

        self.number_of_sensors = 12                         # Number of input sensors
        self.sensor_sentivity = 10                           # Range of sensor

        self.x = 15
        self.y = 15
        self.theta = math.pi
        self.velocity_l = 1
        self.velocity_r = 1
        self.translational_velocity = 1
        self.np_sensor_output = np.zeros(12)


    def initialize_robot(self):
        """Initialize the position of robot to the values
            given to the function. If no values are provided the robot is initialize randomly
            in the legal space"""

        self.x = random.random() * 100
        self.y = random.random() * 100
        i = 1
        while(Environment.check_location(self.x, self.y) == 0):
            self.x = float(random.random() * 25)
            self.y = float(random.random() * 25)
            i += 1

        self.theta = random.random()* math.pi * 2

        return self.x, self.y, self.theta


    def sense_environment(self, x, y):

        temp1 = x
        temp2 = y

        theta = 2 * (math.pi) / self.number_of_sensors
        self.np_sensor_output = np.zeros(self.number_of_sensors);


        for j in range(self.number_of_sensors):
            x = temp1
            y = temp2

            for i in range(self.sensor_sentivity + 1):
                if Environment.check_location(x, y) == 0:
                    self.np_sensor_output[j] = self.sensor_sentivity - i
                    break
                x = x + 0.5 * math.cos(theta * (j + 1))
                y = y + 0.5 * math.sin(theta * (j + 1))


        return self.np_sensor_output

    def motion_model(self, velocity_left, velocity_right):

        self.velocity_l = velocity_left
        self.velocity_r = velocity_right

        theta = self.theta

        self.translational_velocity = (velocity_right + velocity_left)/2
        #theta = math.atan(x, y)

        # Angular Velocity
        omega = (velocity_right - velocity_left)/self.distance_between_wheels

        if (velocity_right != velocity_left):
            radius_of_curv = 1/2 * (velocity_right + velocity_left)/(velocity_right - velocity_left)

            # Instantaneous centre of Curvature
            centre_of_curv = [self.x - radius_of_curv*math.sin(theta), self.y + radius_of_curv*math.cos(theta)]

            # new positions
            self.x = math.cos(omega*TIME_STEP)*(self.x - centre_of_curv[0]) - math.sin(omega * TIME_STEP) * (self.y - centre_of_curv[1]) + centre_of_curv[0]
            self.y = math.sin(omega*TIME_STEP)*(self.x - centre_of_curv[0]) + math.cos(omega * TIME_STEP) * (self.y - centre_of_curv[1]) + centre_of_curv[1]
            self.theta = theta + omega* TIME_STEP
        else:
            self.x = self.x + self.translational_velocity*TIME_STEP * math.cos(self.theta)
            self.y = self.y + self.translational_velocity*TIME_STEP * math.sin(self.theta)

        return (self.x, self.y, self.theta)

    def collision_check(self):
        # if np.max(self.np_sensor_output) >= 9:
        #     self.velocity_r = -self.velocity_r
        #     self.velocity_l = -self.velocity_l
        #     self.motion_model(self.velocity_l, self.velocity_r)
        if Environment.check_location(self.x, self.y) == 0:
            if self.x <= 0 + 1 or (self.x >= 70 and self.x <= 75 + 1):
                self.x = self.x + 2
            elif self.x >= 100 - 1 or (self.x >= 25 - 1 and self.x <= 30):
                self.x  = self.x - 2

            if self.y <= 0 + 1 or (self.y >= 70 and self.y <= 75 + 1):
                self.y = self.y + 2;
            elif self.y >= 100 - 1 or (self.y >= 25 - 1 and self.y <= 30):
                self.y = self.y - 2;

            self.theta = self.theta + math.pi
            self.motion_model(self.velocity_l, self.velocity_r)

    def clean_dirt(self, x, y):
        grid_number_x = math.floor(x/4)
        grid_number_y = math.floor(y/4)
        NP_GRID[grid_number_x, grid_number_y] = 1
        return np.count_nonzero(NP_GRID)






class Neural_Net:

    def __init__(self):
        self.input_nodes = 12
        self.hidden_nodes = 0
        self.output_nodes = 2
        self.np_previous_activation = np.array([0, 0])
        self.np_weight_matrix = None
        self.np_feedback_matrix = None

    def sigmoid(self, x):
        sigmoid = 1/(1 + np.exp(-x))
        return sigmoid

    def transform_weights(self, weight_stream_from_ea):
        """converts weight stream from evolutionary algorithm to
            weight matrix"""

    def feedback_network(self, np_input_signal, np_weight_matrix, np_feedback_matrix):
        """Calculates the final velocities based on the weights obtained
            from evolutionary algorithm

            Things to note:
            input signal is a row vector of size 12
            and weight matrix is a signal with
            weight for each activation node aligned in
            columns  ----> [1 X 13][13 x 2]

            feedback matrix is of size [2 X 2]"""
        self.np_weight_matrix = np_weight_matrix
        self.np_feedback_matrix = np_feedback_matrix

        # print('weight Matrix is' + str(np_weight_matrix))
        # print('feedback matrix is' + str(np_feedback_matrix))
        # print('input signal is' + str(np_input_signal))



        # Adding Bias to input
        bias = np.ones(1)
        np_input_signal = np.concatenate((bias, np_input_signal))


        # Multipying weight Matrix to the input layer
        np_input_layer_1 = np.dot(np_input_signal,  np_weight_matrix)
        #print('linear combination 1: ' +  str(np_input_layer_1))
        # Multiplying feedback weights with the t-1 step activation values
        np_feedback = np.dot(self.np_previous_activation, np_feedback_matrix)
        #print('feedback combination: ' + str(np_feedback))

        np_combined_input =np_input_layer_1 + np_feedback

        # Calculating Sigmoid value for the final otput layer
        np_activation_value = self.sigmoid(np_combined_input)
        #print(np_activation_value)

        self.np_previous_activation = np_activation_value


        return np_activation_value

    def transform_velocity(self):
        """Transforms the neural network output from
            -2 meters/second to 2 meters/second range"""
        np_velocity_vector = (self.np_previous_activation * 4) - 2
        return np_velocity_vector

robo = Robot()
robo.sense_environment(5, 5)