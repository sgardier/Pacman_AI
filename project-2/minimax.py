from pacman_module.game import Agent, Directions
import math


class PacmanAgent(Agent):
    """Pacman agent implementing the minimax algorithm.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()

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
                    returns a legal move using the minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        best_score = - math.inf
        best_action = Directions.STOP
        visited_states = set()
        visited_states.add(self.key(state))
        for successor, action in state.generatePacmanSuccessors():
            visited_states.add(self.key(successor))
            new_score = self.minimax(successor, 1, visited_states)
            visited_states.remove(self.key(successor))
            if new_score > best_score:
                best_score = new_score
                best_action = action
        return best_action

    def minimax(self, state, agent_index, visited_states):
        """Recursively computes the minimax value for the given state.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            agent_index: an index representing the current agent
                            (0 for Pacman, 1 for Ghost).
            visited_nodes: a set to track visited states and avoid cycles.

        Returns:
            The minimax value for the given state.
        """

        if state.isWin() or state.isLose():
            return state.getScore()

        if agent_index == 0:
            best_score = - math.inf
            for successor, _ in state.generatePacmanSuccessors():
                key_state = self.key(state)
                if key_state in visited_states:
                    continue
                visited_states.add(self.key(state))
                ret_score = self.minimax(successor, 1, visited_states)
                visited_states.remove(key_state)
                best_score = max(best_score, ret_score)
            return best_score

        elif agent_index == 1:
            worst_score = + math.inf
            for successor, _ in state.generateGhostSuccessors(1):
                ret_score = self.minimax(successor, 0, visited_states)
                worst_score = min(worst_score, ret_score)
            return worst_score
