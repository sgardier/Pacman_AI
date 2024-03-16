import numpy as np
import math
from pacman_module.util import PriorityQueue
from pacman_module.game import Agent, Directions, manhattanDistance


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transitiion matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        def get_ghost_behavior(ghost):
            fearness_mapping = {'fearless': 0, 'afraid': 1, 'terrified': 3}
            # Default to 0 if the state is not in the dictionary
            return fearness_mapping.get(ghost, 0)

        def possible_actions():
            return [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def fill_transition_matrix_at(i, j, T, walls, ghost_behavior):
            previous_distance = manhattanDistance(position, (i, j))
            for act in possible_actions():
                (k, l) = (i + act[0], j + act[1])
                if not walls[k][l]:
                    T[i, j, k, l] = (2**ghost_behavior
                                     if manhattanDistance(position, (k, l)) >=
                                     previous_distance else 1)
            if np.sum(T[i, j]) != 0:
                T[i, j] = T[i, j] / np.sum(T[i, j])

        width, height = walls.width, walls.height

        T = np.zeros((width, height, width, height))

        ghost_behavior = get_ghost_behavior(self.ghost)

        for i in range(width):
            for j in range(height):
                if not walls[i][j]:
                    fill_transition_matrix_at(i, j, T, walls, ghost_behavior)
        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        width, height = walls.width, walls.height

        # Parameters for the binomial distribution
        n, p = 4, 0.5

        O = np.zeros((width, height))

        for i in range(width):
            for j in range(height):
                if walls[i][j] is False:
                    dist_to_Pacman = manhattanDistance(position, (i, j))
                    # noise calculation
                    z = evidence - dist_to_Pacman + n * p

                    if z < 0:
                        # The probability O[i][j] is set to zero to reflect
                        # that the noisy distance is very unlikely given
                        # the actual distance
                        O[i, j] = 0
                    else:
                        # Calculate the probability using the binomial
                        # distribution formula
                        O[i, j] = (math.comb(n, int(z)) *
                                   (p**z) * ((1 - p)**(n - z))
                                   )
        return O

    def update(
            self, walls, belief,
            evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        width, height = walls.width, walls.height

        predicted_belief = np.tensordot(belief, T, axes=([0, 1], [0, 1]))

        updated_belief = np.multiply(predicted_belief, O)

        normalize_update_belief = updated_belief / \
            np.sum(np.sum(updated_belief))

        return normalize_update_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls, beliefs[i],
                    evidences[i], position
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        def select_ghost_to_eat(beliefs, eaten, position):
            ghost_coord = [np.unravel_index(
                np.argmax(belief), belief.shape) for belief in beliefs]

            ghost_dist = [
                float('inf') if is_eat else manhattanDistance(
                    position, belief_coord) for belief_coord, is_eat in zip(
                    ghost_coord, eaten)]

            ghost_to_eat = np.argmin(ghost_dist)
            belief_ghost_pos = ghost_coord[ghost_to_eat]

            return ghost_to_eat, belief_ghost_pos

        def calculate_weighted_average_position(ghost_matrix):
            (width, height) = ghost_matrix.shape
            x, y = 0, 0

            for i in range(width):
                for j in range(height):
                    x += ghost_matrix[i, j] * i
                    y += ghost_matrix[i, j] * j

            return [round(x), round(y)]

        def a_star_search(
                position,
                ghost_average_pos,
                walls,
                belief_ghost_pos):

            def generateSuccessors(position):
                successors = []
                actions = []

                possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                action_mapping = {
                    (1, 0): Directions.EAST,
                    (-1, 0): Directions.WEST,
                    (0, 1): Directions.NORTH,
                    (0, -1): Directions.SOUTH
                }

                for act in possible_actions:
                    new_position = (position[0] + act[0], position[1] + act[1])

                    if not walls[new_position[0]][new_position[1]]:
                        action = action_mapping.get(act, None)
                        if action is not None:
                            successors.append(new_position)
                            actions.append(action)

                return zip(successors, actions)

            path = []
            fringe = PriorityQueue()
            fringe.push((position, path, 0), 0)
            closed = set()

            if position == (ghost_average_pos[0], ghost_average_pos[1]) or \
                    walls[ghost_average_pos[0]][ghost_average_pos[1]]:
                ghost_average_pos[0] = belief_ghost_pos[0]
                ghost_average_pos[1] = belief_ghost_pos[1]

            while not fringe.isEmpty():
                _, (PacmanPos, path, g) = fringe.pop()

                if (PacmanPos[0], PacmanPos[1]) == (
                        int(ghost_average_pos[0]), int(ghost_average_pos[1])):
                    if len(path) == 0:
                        return Directions.STOP

                    return path[0]

                if (PacmanPos[0], PacmanPos[1]) in closed:
                    continue
                else:
                    closed.add((PacmanPos[0], PacmanPos[1]))

                for successor, action in generateSuccessors(
                        (PacmanPos[0], PacmanPos[1])):

                    # compute the cost to go from current to successor
                    cost = g + 1
                    # compute the forward cost
                    h = manhattanDistance(
                        (successor[0], successor[1]), belief_ghost_pos)
                    # compute the total cost for successor
                    f = cost + h + g
                    fringe.push((successor, path + [action], cost), f)

            return path

        ghost_to_eat, belief_ghost_pos = select_ghost_to_eat(
            beliefs, eaten, position)

        ghost_average_pos = calculate_weighted_average_position(
            beliefs[ghost_to_eat])

        return a_star_search(
            position,
            ghost_average_pos,
            walls,
            belief_ghost_pos)

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
