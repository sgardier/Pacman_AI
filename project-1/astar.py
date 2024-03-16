from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """
    pacmanPos = state.getPacmanPosition()
    food = state.getFood()
    capsules = state.getCapsules()
    return (
        pacmanPos,
        food,
        tuple(capsules)
    )


class PacmanAgent(Agent):
    """Pacman agent based on A* search."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def cost(self, state, capsulesPreviousState):
        """Given a state, compute the cost, if pacman ate a capsule
            we augment its cost of 5, in all case we add 1 since we decided
            that a move cost 1

        Arguments:
            state: a game state
            capsulesPreviousState: the list of the positions of the capsules

        Returns:
            The cost (integer)
        """
        cost = 1
        if state.getPacmanPosition() in capsulesPreviousState:
            cost += 5
        return cost

    def h(self, state):
        """Given a state, compute the distance with the furthest
           food from the position of Pacman.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A manhattan distance (integer)
        """
        food_list = state.getFood().asList()
        if len(food_list) == 0:
            return 0
        pacman_position = state.getPacmanPosition()
        distances = [manhattanDistance(pacman_position, food)
                     for food in food_list]
        return max(distances)

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        path = []
        fringe = PriorityQueue()
        fringe.push((state, path, 0), 0)
        closed = set()

        while not fringe.isEmpty():
            priority, (current, path, g) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                # compute the cost to go from current to successor
                cost = self.cost(successor, current.getCapsules())
                # compute the forward cost
                h = self.h(successor)
                # compute the total cost for successor
                f = g + cost + h
                fringe.push((successor, path + [action], g + cost), f)

        return path
