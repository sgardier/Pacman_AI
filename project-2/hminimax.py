from pacman_module.game import Agent, Directions
from pacman_module.util import Queue, manhattanDistance
import math


class PacmanAgent(Agent):
    """Pacman agent implementing a heuristic minimax algorithm.

    Attributes:
        max_depth (int): Maximum depth to explore in the search tree.
        positionsHistory (dict): Dictionary to track
                the number of times Pacman has visited each position.
        ghost_acceptable_dist (int): Acceptable distance from Ghost to Pacman.
    """

    def __init__(self):
        super().__init__()
        self.max_depth = 2
        self.positions_history = {}
        self.ghost_acceptable_dist = 12

    def key(self, state):
        """Returns a key that uniquely identifies a Pacman game state.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A hashable key tuple.
        """

        pacmanPos = state.getPacmanPosition()
        ghostPos = state.getGhostPosition(1)
        food = state.getFood()
        return (pacmanPos, ghostPos, food)

    def get_action(self, state):
        """Given a Pacman game state,
            returns a legal move using the heuristic minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        best_score = -math.inf
        best_action = Directions.STOP
        best_successor = state
        visited_states = set()
        visited_states.add(self.key(state))

        for successor, action in state.generatePacmanSuccessors():
            visited_states.add(self.key(successor))
            new_score = self.hminimax(
                successor,
                agent_index=1,
                visited_states=visited_states,
                depth=0,
                evaluated_position=successor.getPacmanPosition(),
            )
            visited_states.remove(self.key(successor))
            if new_score > best_score:
                best_score = new_score
                best_action = action
                best_successor = successor

        self.positions_history[best_successor.getPacmanPosition()] = \
            self.positions_history.get(best_successor.getPacmanPosition(), 0)\
            + 1

        return best_action

    def cutoff_test(self, state, depth):
        """Checks if the search should be
            cut off at the current state and depth.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth in the search tree.

        Returns:
            True if the search should be cut off, False otherwise.
        """

        ghost_position = state.getGhostPosition(1)
        pacman_position = state.getPacmanPosition()

        if (
            manhattanDistance(pacman_position, ghost_position) >
            self.ghost_acceptable_dist
        ):
            return (state.isWin() or
                    state.isLose() or
                    depth >= self.max_depth
                    )
        else:
            return (state.isWin() or
                    state.isLose() or
                    depth >= self.max_depth + 1)

    def shortest_foods_path(self, walls_matrix, foods_matrix, start_position):
        """Computes the smallest distance to eat all remaining foods using BFS.

        Arguments:
            walls_matrix: a boolean matrix representing the walls in the game.
            foods_matrix: a boolean matrix representing the remaining foods.
            start_position: the starting position for the search.

        Returns:
            The shortest distance to walk to eat all the foods.
        """

        def nearest_food_distance(walls_matrix, foods_list, start_position):

            def is_valid_move(x, y):
                return (
                    0 <= x < walls_matrix.width
                    and 0 <= y < walls_matrix.height
                    and not walls_matrix[x][y]
                )

            fringe = Queue()
            fringe.push((start_position, 0))
            closed = set()

            while not fringe.isEmpty():
                (x, y), distance = fringe.pop()

                if (x, y) in foods_list:
                    return distance, (x, y)

                for dx, dy in [(1, 0), (- 1, 0), (0, 1), (0, - 1)]:
                    nx, ny = x + dx, y + dy
                    if is_valid_move(nx, ny) and (nx, ny) not in closed:
                        fringe.push(((nx, ny), distance + 1))
                        closed.add((nx, ny))
            return math.inf, start_position

        total_distance = 0
        foods_list = [
            (i, j)
            for i, row in enumerate(foods_matrix)
            for j, value in enumerate(row)
            if value
        ]

        curr_position = start_position
        while foods_list:
            curr_distance, curr_position = \
                nearest_food_distance(
                    walls_matrix, foods_list, curr_position)
            if curr_distance == math.inf:
                return total_distance
            else:
                total_distance += curr_distance
                foods_list.remove(curr_position)

        return total_distance

    def eval(self, state, evaluated_position):
        """Evaluates the current state based on the heuristic.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            evaluated_position: the position of Pacman which we evaluate

        Returns:
            The heuristic evaluation of the state.
        """

        if state.isWin() or state.isLose():
            return state.getScore()
        h = state.getScore()
        h -= self.shortest_foods_path(state.getWalls(),
                                      state.getFood(),
                                      state.getPacmanPosition())
        h -= math.exp(self.positions_history.get(evaluated_position, 0))
        return h

    def hminimax(self,
                 state,
                 agent_index,
                 visited_states,
                 depth,
                 evaluated_position
                 ):
        """Recursively computes
            the heuristic minimax value for the given state.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            agent_index: the index of the current agent
                            (0 for Pacman, 1 for Ghost).
            visited_states: a set of visited states to avoid cycles
                            in the current branch of the search tree.
            depth: the current depth in the search tree.
            evaluated_position: the new position for
                                which we evaluate the result.

        Returns:
            The heuristic minimax value of the state.
        """
        if self.cutoff_test(state, depth):
            return self.eval(state, evaluated_position)

        if agent_index == 0:
            best_score = -math.inf
            for successor, _ in state.generatePacmanSuccessors():
                key_state = self.key(successor)
                if key_state not in visited_states:
                    visited_states.add(key_state)
                    score = self.hminimax(
                        state=successor,
                        agent_index=1,
                        visited_states=visited_states,
                        depth=depth + 1,
                        evaluated_position=evaluated_position)
                    visited_states.remove(key_state)
                    best_score = max(best_score, score)
            return best_score
        else:
            worst_score = math.inf
            for successor, _ in state.generateGhostSuccessors(1):
                score = self.hminimax(
                    state=successor,
                    agent_index=0,
                    visited_states=visited_states,
                    depth=depth + 1,
                    evaluated_position=evaluated_position)
                worst_score = min(worst_score, score)
            return worst_score
