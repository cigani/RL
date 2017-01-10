import sys

import pylab as plb
import numpy as np
import mountaincar
import itertools
import h5py
from datetime import datetime


class DummyAgent:
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, eta=0.1, gamma=0.95, lam=0.8,
                 initial_epsilon=0.1, min_epsilon=0.0, half_life=1.0,
                 initial_temperature=1.0, min_temperature=0.01,
                 temperature_half_life=1.0, neurons=10, time=100,
                 dt=0.01, actions=3, n_steps=5000, n_episodes=350,
                 run_type="Default", explore_temp=False, explore_lam=False,
                 explore_both=False, explore_weights=False, weights=.05):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        # Learning rate
        self.eta_ = eta
        # Reward Factor
        self.gamma_ = gamma
        # Decay Eligibility
        self.lambda_ = lam

        # Choice of Random Action or Not
        self.initial_epsilon_ = initial_epsilon
        self.min_epsilon_ = min_epsilon
        self.epsilon_half_life_ = half_life

        # Exploration vs Exploitation parameter
        self.initial_temperature_ = initial_temperature
        self.min_temperature_ = min_temperature
        self.temperature_half_life_ = temperature_half_life

        # Neuron Centers
        self.neuron = neurons
        self.neuron_count = self.neuron ** 2
        _x_space_, self.x_centers_distance = np.linspace(-150, 30, neurons,
                                                         retstep=True)
        _x_d_space_, self.phi_centers_distance = np.linspace(-15, 15, neurons,
                                                             retstep=True)
        self.centers = np.array(list(itertools.product(_x_space_,
                                                       _x_d_space_)))
        self.x_sigma = self.x_centers_distance ** 2
        self.x_d_sigma = self.phi_centers_distance ** 2

        # Activity / State Parameters
        self.activity = {"Right": 0, "Left": 1, "Neutral": 2}
        self.action_index_ = {"1": 0, "-1": 1, "0": 2}
        self.last_action = None
        self.action = 0
        self.old_state = None
        self.state = [self.mountain_car.x, self.mountain_car.x_d]
        self.q_weights = 0.0
        self.hold = np.zeros(3)

        # Trace Memory
        self.e = np.zeros((self.neuron_count, actions))
        self.weights = 0.00001 * np.random.rand(self.neuron_count, actions)
        if explore_weights:
            self.weights = np.ones(self.neuron_count, actions) * weights

        # Time step for Simulation
        self.time = time
        self.dt = dt
        self.n_steps = n_steps
        self.n_episodes = n_episodes

        # Exploration
        self.explore_temp = explore_temp
        self.explore_lam = explore_lam
        self.explore_both = explore_both

        # Save Data
        self.filename = "{0}-{1}s.hdf5".format(run_type,
                                               datetime.now().strftime(
                                                   '%m-%d-%H.%M.%S'))

    def visualize_trial(self, visual=False):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        # H5 Data Sets #
        h5data = h5py.File(self.filename, 'w')
        h5data.create_group('episode_rewards')
        h5data.create_group('x_data')
        h5data.create_group('x_dot_data')
        h5data.create_group('force_data')
        h5data.create_group('q_data')

        episode_rewards = np.zeros(self.n_episodes)
        x_data = np.zeros(self.n_steps)
        x_dot_data = np.zeros(self.n_steps)
        force_data = np.zeros(self.n_steps)
        q_data = np.zeros((self.n_steps, 3))

        # prepare for the visualization
        if visual:
            plb.ion()
            mv = mountaincar.MountainCarViewer(self.mountain_car)
            mv.create_figure(self.n_steps, self.n_steps)
            mv.update_figure()
            plb.draw()
            plb.show()
            plb.pause(0.0001)

        for _ in np.arange(self.n_episodes):
            self.mountain_car.reset()
            if self.explore_temp:
                self.initial_temperature_ = self._time_decay(
                    self.initial_temperature_, _)
            if self.explore_lam:
                self.lambda_ = self._time_decay(self.lambda_, _)
            if self.explore_both:
                self.initial_temperature_ = self._time_decay(
                    self.initial_temperature_, _)
                self.lambda_ = self._time_decay(self.lambda_, _)
            for n in range(self.n_steps):
                x_data[n] = self.mountain_car.x
                x_dot_data[n] = self.mountain_car.x_d
                force_data[n] = self.mountain_car.F
                q_data[n] = self.hold
                self._learn()
                if self.mountain_car.R > 0.0:
                    np.insert(episode_rewards, _, self.mountain_car.t)
                    h5data['x_data'].create_dataset(
                        'x_data_{}'.format(_), (self.n_steps, 1),
                        maxshape=(self.n_steps, 1), data=x_data,
                        chunks=True, compression="gzip")
                    h5data['x_dot_data'].create_dataset(
                        "x_dot_data_{}".format(_), (self.n_steps, 1),
                        maxshape=(self.n_steps, 1), data=x_dot_data,
                        chunks=True, compression="gzip")
                    h5data['force_data'].create_dataset(
                        "force_data_{}".format(_), (self.n_steps, 1),
                        maxshape=(self.n_steps, 1), data=force_data,
                        chunks=True, compression="gzip")
                    h5data['q_data'].create_dataset(
                        "q_data_{}".format(_), (self.n_steps, 3),
                        maxshape=(self.n_steps, 3), data=q_data,
                        chunks=True, compression="gzip")

                    print("\rreward obtained at t = ", self.mountain_car.t)
                    break
        h5data['episode_rewards'].create_dataset('episode_reward',
                                                 (self.n_episodes, 1),
                                                 maxshape=(self.n_episodes, 1),
                                                 data=episode_rewards,
                                                 chunks=True,
                                                 compression="gzip")

    def _learn(self):
        self._action_choice()

    def _input_layer(self, neuron_index):
        rj = np.exp((-(neuron_index[0] - self.state[0]) ** 2) /
                    self.x_sigma - ((neuron_index[1] - self.state[1]) ** 2) /
                    self.x_d_sigma)
        return rj

    def _output_layer(self, action_index):
        self.q_weights = 0.0
        for n in np.arange(self.neuron_count):
            self.q_weights += (self.weights[n][action_index] *
                               self._input_layer(self.centers[n]))
        self.hold[action_index] = self.q_weights
        return self.q_weights

    def _update_eligibility(self):
        """
        Eligibility updates using the SARSA protocol.
        """
        action = self.action_index_["{}".format(self.last_action)]
        self.e *= self.lambda_ * self.gamma_
        self.e[self.old_index, action] += \
            self._input_layer(self.centers[self.old_index])

    def _update_weights(self):
        """
        Weight updates using the SARSA protocol.
        """
        self._td_error()
        self.weights += self.dirac * self.eta_ * self.e
        self.weights = np.clip(self.weights, -1, 1)

    def _td_error(self):
        reward = self.mountain_car.R
        self.dirac = (reward - (
            self._output_layer(self.last_action) -
            self.gamma_ * self._output_layer(self.action)))

    def _action_choice(self):
        """
        Choose the next action based on the current Q-values.
        Determined by soft-max rule.
        Cumulative probabilities for quick implementation.
        """

        c_prob_left = self._soft_max_rule(self.activity["Left"])
        c_prob_right = c_prob_left + self._soft_max_rule(
            self.activity["Right"])

        test_value = np.random.rand()

        if test_value < c_prob_left:
            self._update_state(-1)
        elif test_value < c_prob_right:
            self._update_state(1)
        else:
            self._update_state(0)

    def _soft_max_rule(self, action):
        """
        Soft-max algorithm
        """
        probability = (np.exp(self._output_layer(action)
                              / self.initial_temperature_)
                       / self._all_actions())
        return probability

    def _all_actions(self):
        """
        For the denominator of the soft-max algorithm
        """
        total_q = 0.0
        for action in self.action_index_.values():
            total_q += np.exp(self._output_layer(action)
                              / self.initial_temperature_)
        return total_q

    def _update_state(self, command):
        self.last_action = self.action
        self.action = command
        self.old_x = self.mountain_car.x
        self.old_x_d = self.mountain_car.x_d
        self.old_state = [self.old_x, self.old_x_d]
        self.old_index = self._get_index(self.old_state)

        self.mountain_car.apply_force(self.action)
        self.mountain_car.simulate_timesteps(self.time, self.dt)

        self.x = self.mountain_car.x
        self.x_d = self.mountain_car.x_d
        self.state = [self.x, self.x_d]
        self.new_index = self._get_index(self.state)

        self._update_eligibility()
        self._update_weights()

    def _get_index(self, state):
        return np.sum(np.square(np.abs(self.centers - state)), 1).argmin()
        # Alternative Method. It's MUCH more expensive though
        # (np.array([np.linalg.norm(m + c)
        #               for (m, c) in
        #               np.abs(self.centers - state)]).argmin())

    def _time_decay(self, val, time):
        mada = (self.n_episodes * 0.33) / np.log(val + 0.00000000001)
        return val * np.exp(-time / mada)


if __name__ == "__main__":
    d = DummyAgent(explore_lam=True, run_type="explore_lam", n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(explore_temp=True, run_type="explore_temp", n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(explore_both=True, run_type="explore_both", n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(explore_weights=True, weights=0.0,
                   run_type="zero_weight", n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(explore_weights=True, weights=1.0, run_type="one_weight",
                   n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(initial_temperature=0.0, run_type="zero_temp",
                   n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(initial_temperature=10e5, run_type="inf_temp",
                   n_episodes=150)
    d.visualize_trial()
    d = DummyAgent(lam=0.0, run_type="zero_lambda", n_episodes=150)
    d.visualize_trial()
