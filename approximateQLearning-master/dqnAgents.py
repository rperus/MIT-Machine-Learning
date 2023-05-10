from collections import deque

import numpy
from keras import Sequential
from keras.layers import Dense, Convolution2D, Flatten, np
from keras.optimizers import Adam

from game import *
from graphicsDisplay import getFrame, saveFrame
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,keras
from keras.optimizers import SGD
from skimage.transform import resize
import tensorflow as tf

PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']


class PacmanDQNAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.state_size = 294
        #self.state_size = 1320
        self.action_size = 5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.count = 0
        self.init = 0
        self.epsilon = 1.0
        self.alpha = 0.001
        #self.alpha = 1e-6
        self.discount = 0.95
        self.batch_size = 35
        self.alpha_decay = 0.01
        self.frame_width = 85
        self.frame_height = 85

        self.memory = deque(maxlen=20000)
        self.model = self._build_model()



    # def getQValue(self, state, action):
    #     """
    #       Returns Q(state,action)
    #       Should return 0.0 if we have never seen a state
    #       or the Q node value otherwise
    #     """
    #     "*** YOUR CODE HERE ***"
    #     #return self.QValueCounter[(state, action)]
    #
    #
    # def computeValueFromQValues(self, state):
    #     """
    #       Returns max_action Q(state,action)
    #       where the max is over legal actions.  Note that if
    #       there are no legal actions, which is the case at the
    #       terminal state, you should return a value of 0.0.
    #     """
    #     "*** YOUR CODE HERE ***"
    #     if not self.getLegalActions(state): return 0
    #
    #     best_action = self.computeActionFromQValues(state)
    #     return self.getQValue(state, best_action)
    #
    # def computeActionFromQValues(self, state):
    #     """
    #       Compute the best action to take in a state.  Note that if there
    #       are no legal actions, which is the case at the terminal state,
    #       you should return None.
    #     """
    #     "*** YOUR CODE HERE ***"
    #     if not self.getLegalActions(state): return None
    #
    #     best_action = None
    #     best_value = float('-inf')
    #     for action in self.getLegalActions(state):
    #         if self.getQValue(state, action) > best_value:
    #             best_value = self.getQValue(state, action)
    #             best_action = action
    #     return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        # if 'Stop' in legalActions:
        #     legalActions.remove('Stop')

        action = None
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return action  # Terminal State, return None

        self.image = getFrame()
        self.image = np.array(self.image)
        self.image = resize(self.image, (self.frame_width, self.frame_height))
        self.image = np.reshape(self.image, [1,self.frame_height, self.frame_width,3])
        self.image = np.uint8(self.image)

        #print 'Epsilon value: ', self.epsilon
        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            #action = self.computeActionFromQValues(state)  # Exploit
            #state_matrix = self.getStateMatrices(state)
            #state_matrix = np.reshape(np.array(state_matrix), [1, self.state_size])
            #state_matrix = np.reshape(state_matrix, (1, self.frame_width, self.frame_height))

            act_values = self.model.predict(self.image)
            action = PACMAN_ACTIONS[(np.argmax(act_values[0]))] # returns action

            if action not in legalActions:
                action = 'Stop'
                #action = random.choice(legalActions)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.nextImage = getFrame()
        self.nextImage = numpy.array(self.nextImage)
        self.nextImage = resize(self.nextImage, (self.frame_width, self.frame_height))
        self.nextImage = np.reshape(self.nextImage, [1, self.frame_height, self.frame_width, 3])
        self.nextImage = np.uint8(self.nextImage)
        #new code
        if not self.getLegalActions(state):
            done = True
        else:
            done = False

        # state_matrix = self.getStateMatrices(state)
        # nextState_matrix = self.getStateMatrices(nextState)
        # state_matrix = np.reshape(state_matrix, [1, self.state_size])
        # nextState_matrix = np.reshape(nextState_matrix, [1, self.state_size])

        self.remember(self.image, action, reward, self.nextImage, done)

        # entrenamos la red neuronal solo mientras estamos en entrenamiento
        #if self.episodesSoFar < self.numTraining:
        #if self.episodesSoFar < self.numTraining:
        if len(self.memory) > 2*self.batch_size:
            self.replay(self.batch_size)

    # def getPolicy(self, state):
    #     return self.computeActionFromQValues(state)
    #
    # def getValue(self, state):
    #     return self.computeValueFromQValues(state)


    # def final(self, state):
    #     "Called at the end of each game."
    #     # call the super-class final method
    #     ReinforcementAgent.final(self, state)
    #
    #     state_matrix = self.getStateMatrices(state)
    #     nextState_matrix = self.getStateMatrices(self.lastState)
    #     state_matrix = np.reshape(state_matrix, [1, self.state_size])
    #     nextState_matrix = np.reshape(nextState_matrix, [1, self.state_size])
    #
    #     self.remember(state_matrix, action, reward, nextState_matrix, done)
    #
    #     # did we finish training?
    #     if self.episodesSoFar == self.numTraining:
    #         # you might want to print your weights here for debugging
    #         "*** YOUR CODE HERE ***"
    #         pass

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(48, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        model = Sequential()
        #STATE_LENGTH = self.state_size
        FRAME_WIDTH = self.frame_width
        FRAME_HEIGHT = self.frame_height
        STATE_LENGTH = 3
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
        #                        input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
                                 input_shape=(FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))


        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = PACMAN_ACTIONS.index(action)
        self.memory.append((state, action_index, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        for state, action_index, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action_index] = reward if done else reward + self.discount * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=1)

        if self.epsilon > self.epsilon_min:
            #None
            self.epsilon *= self.epsilon_decay


    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height = state.data.layout.width, state.data.layout.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        #print 'observation: ', observation
        #observation = np.swapaxes(observation, 0, 2)

        #print 'observation: ', observation
        return observation
