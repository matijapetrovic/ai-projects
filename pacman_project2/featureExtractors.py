# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"
import math

from game import Directions, Actions
import util


class FeatureExtractor:
    def getFeatures(self, agent, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, agent, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, agent, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


def get_ghosts_3_away(agent, state, pos, ghosts):
    sum = 0
    for g in ghosts:
        ghost_pos = state.getAgentPosition(g)
        dist = agent.getMazeDistance(pos, ghost_pos)
        if dist <= 5:
            sum += 1
    return sum

def get_min_opponents_distance(agent, state, pos, ghosts):
    min_distance = 10000
    for g in ghosts:
        ghost_pos = state.getAgentPosition(g)
        ghost_state = state.getAgentState(g)
        dist = agent.getMazeDistance(pos, ghost_pos)
        if dist <= min_distance:  # zasto ne radi ako ovde stavim isPacman ?
            min_distance = dist
    return min_distance

def check_opponents( state, ghosts):
    sum = 0
    for g in ghosts:
        ghost_state = state.getAgentState(g)
        if ghost_state.isPacman:
            sum += 1
    return sum

def get_pacmans_1_step_away(agent, state, pos, ghosts):
    sum = 0
    for g in ghosts:
        ghost_pos = state.getAgentPosition(g)
        ghost_state = state.getAgentState(g)
        dist = agent.getMazeDistance(pos, ghost_pos)
        if dist <= 5 and ghost_state.isPacman:
            sum += 1
    return sum


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, agent, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = agent.getFood(state)
        walls = state.getWalls()
        ghosts = [state.getAgentPosition(g) for g in agent.getOpponents(state)]

        features = util.Counter()
        features["bias"] = 1.0


        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(agent.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pos = (next_x, next_y)
        num_carrying = state.getAgentState(agent.index).numCarrying
        if num_carrying > 0:
            dist = agent.getMazeDistance(agent.start_pos, (next_x, next_y))
            features["run-home"] = float(dist) / (walls.width * walls.height) * num_carrying

        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(state.getAgentPosition(g), walls) for g in agent.getOpponents(state) if state.getAgentState(g).scaredTimer == 0)

        ghost_distances = [agent.getMazeDistance(pos, g) for g in ghosts]

        closest_ghost = max(min(ghost_distances), 0.5)
        features["dist-to-closest-ghost"] = float(closest_ghost) / (walls.width * walls.height)

        capsules = [agent.getMazeDistance(pos, c) for c in agent.getCapsules(state)]
        if len(capsules) > 0:
            features["dist-to-closest-capsule"] = float(min(capsules)) / (walls.width * walls.height)

        scared_ghost_distances = [agent.getMazeDistance(pos, state.getAgentPosition(g)) for g in agent.getOpponents(state) if state.getAgentState(g).scaredTimer > 0 and not state.getAgentState(g).isPacman]
        if len(scared_ghost_distances) > 0:
            features["dist-to-closest-scared-ghost"] = float(min(scared_ghost_distances)) / (walls.width * walls.height)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
