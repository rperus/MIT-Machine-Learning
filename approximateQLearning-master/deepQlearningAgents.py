from game import *
from learningAgents import ReinforcementAgent
import random,util,math


class DeepQLearningAgent(ReinforcementAgent):
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
        # TODO inicializar la red neuronal (o cargarla de memoria)
        # TODO inicializar la memoria o lo que sea?

        # self.QValueCounter = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(state, action)]

    def getQValues(self, state):
        """
            Returns collection of Qs(state)
        """
        # TODO pasar el

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return 0

        best_action = self.computeActionFromQValues(state)
        return self.getQValue(state, best_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None

        best_action = None
        best_value = float('-inf')

        qvalues = self.getQValues(state)
        for value in qvalues:
            if value > best_value: # TODO comprobar que la accion es legal
                best_value = value
        return best_action

        # for action in self.getLegalActions(state):
        #     if self.getQValue(state, action) > best_value:
        #         best_value = self.getQValue(state, action)
        #         best_action = action
        # return best_action

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
        action = None
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None  # Terminal State, return None

        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            action = self.computeActionFromQValues(state)  # Exploit

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
        # TODO coger estado actual y guardarlo en la memoria
        # TODO recuperar estados de la memoria y entrenar la red neuronal


        # best_action = self.computeActionFromQValues(nextState)
        # self.QValueCounter[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action) +
        #                                        self.alpha * (reward + self.discount * self.getQValue(nextState,
        #                                                                                              best_action)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)