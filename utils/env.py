"""
Environment file.

Keep this file general.
"""
import numpy as np



class environment:
    def __init__(self):
        self.reset()


    def reset(self):
        """
        Reset the environment
        :return:
        """
        self.state, self.actions = envState()
        self.num_act = self.actions.shape[0]
        self.total_reward = 0

        # TODO: introduce global history? Lists or np.arrays?
        self.reward_history = np.array([])
        self.action_history = np.array([])
        self.info_history = np.array([]) # For other random stuff

        self.ep_ended = False

    def perform_action(self, action):
        assert(action<self.num_act, "Not valid action selected")
        new_reward = 0

        pass

        self.total_reward += new_reward

        # TODO: update history

        return [self.state, new_reward, self.ep_ended, self.info_history]

    def is_episode_finished(self):
        return self.ep_ended


    def render(self):
        """Not sure if necessary..."""
        self.state.render()

    def action_space(self):
        """
        Return the possible actions
        """
        return self.actions

    def sample_action(self):
        return np.random.choice(self.num_act)

    def get_state(self):
        return self.state


class envState:
    def __init__(self):
        pass

    def return_current_pred(self):
        pass

    def render(self):
        pass

