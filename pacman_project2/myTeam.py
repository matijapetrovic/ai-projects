# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game

#################
# Team creation #
#################
from learningAgents import ValueEstimationAgent


def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DefensiveAgent', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex, **args), eval(second)(secondIndex, **args)]

##########
# Agents #
##########

# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
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
    def __init__(self, index, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, index, **args)
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        possibleStateQValues = util.Counter()
        for action in self.getLegalActions(state, self.index):
            possibleStateQValues[action] = self.getQValue(state, action)

        if len(possibleStateQValues) > 0:
            return possibleStateQValues[possibleStateQValues.argMax()]
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        possibleStateQValues = util.Counter()
        possibleActions = self.getLegalActions(state, self.index)
        if len(possibleActions) == 0:
            return None

        for action in possibleActions:
            possibleStateQValues[action] = self.getQValue(state, action)

        best_actions = []
        best_value = possibleStateQValues[possibleStateQValues.argMax()]

        for action, value in possibleStateQValues.items():
            if value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

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
        legalActions = self.getLegalActions(state, self.index)
        action = None

        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
        reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, index, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        QLearningAgent.__init__(self, index, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, index, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, index, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qValue = 0.0
        features = self.featExtractor.getFeatures(self, state, action)
        if self.isInTesting():
            print(features)
        for key in features.keys():
            qValue += (self.weights[key] * features[key])
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(self, state, action)
        diff = self.alpha * ((reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        print("END WEIGHTS:")
        print(self.weights)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class DummyAgent(ApproximateQAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def __init__(self, index, extractor="SimpleExtractor", **args):
      ApproximateQAgent.__init__(self, index, extractor, **args)
      self.weights["bias"] = 1.0
      self.weights["closest-food"] = -0.5
      self.weights["eats-food"] = 1.0
      self.weights["run-home"] = -15.0
      self.weights["#-of-ghosts-1-step-away"] = -10.0
      self.weights["dist-to-closest-ghost"] = 0.5
      self.weights["dist-to-closest-capsule"] = -1.0
      self.weights["dist-to-closest-scared-ghost"] = -2.0

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start_pos = gameState.getAgentPosition(self.index)
    ReinforcementAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    """
    return PacmanQAgent.getAction(self, gameState)


class DefensiveAgent(CaptureAgent):
    def __init__(self, index, **args):
        CaptureAgent.__init__(self, index)

    def getValue(self, state, opponent):
        my_pos = state.getAgentPosition(self.index)
        opponent_pos = state.getAgentPosition(opponent)
        return self.getMazeDistance(my_pos, opponent_pos)


    def minimax(self, state, opponent, next_player=-1, depth=5):
        for action in state.getLegalActions(self.index):
            x, y = state.getAgentPosition(self.index)
            dx, dy = Actions.directionToVector(action)
            pos = int(x + dx), int(y + dy)

            opp_pos = state.getAgentPosition(opponent)
            if self.getMazeDistance(opp_pos, pos) == 0:
                return 0, action
        if depth == 0:
            val = self.getValue(state, opponent)
            return val, None

        next_player = self.generate_next_player(next_player, opponent)
        if self.index != next_player:
            return self.max_value(state, depth, opponent, next_player)
        else:
            return self.min_value(state, depth, opponent, next_player)


    def min_value(self, state, depth, opponent, next_player):
        v = math.inf
        legalActions = state.getLegalActions(next_player)
        retAction = None
        for action in legalActions:
            newState = state.generateSuccessor(next_player, action)
            eval, _ = self.minimax(newState, opponent, next_player, depth - 1)
            if eval < v:
                v = eval
                retAction = action
        return v, retAction


    def max_value(self, state, depth, opponent, next_player):
        v = -math.inf
        retAction = None
        legalActions = state.getLegalActions(next_player)
        for action in legalActions:
            newState = state.generateSuccessor(next_player, action)
            eval, _ = self.minimax(newState, opponent, next_player, depth - 1)
            if eval > v:
                v = eval
                retAction = action
        return v, retAction


    def generate_next_player(self, current_next_player, opponent):
        if current_next_player == opponent or current_next_player == -1:
            return self.index
        return opponent


    def calculate_closest_opponent_distance(self, ghosts, state, pos):
        min_distance = 10000
        opponent_index = -1
        for g in ghosts:
            ghost_pos = state.getAgentPosition(g)
            ghost_state = state.getAgentState(g)
            dist = self.getMazeDistance(pos, ghost_pos)
            if dist <= min_distance and ghost_state.isPacman:
                min_distance = dist
                opponent_index = g
        return min_distance, opponent_index



    def get_closest_opponent_distance(self, state, action):
        ghosts = self.getOpponents(state)
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        return self.calculate_closest_opponent_distance(ghosts, state, (next_x, next_y))


    def minimax_allowed(self, state, legalActions, distance=5):
        for action in legalActions:
            min_opponent_distance, opponent_index = self.get_closest_opponent_distance(state, action)
            if min_opponent_distance <= distance:
                return True, opponent_index
        return False, None

    def getAction(self, state):
        legalActions = state.getLegalActions(self.index)
        bestAction = None

        # samo ako je najblizi protivnik pacman i na odredjenoj je udaljenosti uradi minimax
        if len(legalActions) > 0:
            doMinimax, opponent_index = self.minimax_allowed(state, legalActions)
            if doMinimax:
                minimax_value, bestAction = self.minimax(state, opponent_index)
            else:
                walls = state.getWalls()
                middle = (walls.width / 2, walls.height / 2)
                if self.red:
                    middle = (middle[0] - 1, middle[1] - 1)
                min_dist = math.inf
                for action in legalActions:
                    x, y = state.getAgentPosition(self.index)
                    dx, dy = Actions.directionToVector(action)
                    pos = int(x + dx), int(y + dy)
                    dist = self.getMazeDistance(middle, pos)
                    if dist < min_dist:
                        min_dist = dist
                        bestAction = action
        return bestAction


