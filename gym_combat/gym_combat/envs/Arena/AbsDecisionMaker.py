import abc

from gym_combat.gym_combat.envs.Arena.CState import State
from gym_combat.gym_combat.envs.Common.constants import AgentAction, AgentType


class AbsDecisionMaker(metaclass=abc.ABCMeta):

    def update_context(self, state: State, action: AgentAction, new_state: State, reward, is_terminal):

        pass

    def get_action(self, state: State)-> AgentAction:

        pass

    def set_initial_state(self, state: State):

        pass

    def type(self)-> AgentType:

        pass

    def get_epsolon(self):

        return -1

    def save_model(self, number_of_rounds):

        pass

    def get_cover(self):

        return None

    def reset_cover(self):

        pass
