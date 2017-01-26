import itertools
from datetime import datetime

import numpy as np
import pylab as plb

import dataSets
import mountaincar


class DummyAgent:
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, eta=0.1, gamma=0.95, lam=0.8,
                 initial_epsilon=0.1, min_epsilon=0.0, half_life=1.0,
                 initial_temperature=1.0, min_temperature=0.01,
                 temperature_half_life=1.0, neurons=10, time=100,
                 dt=0.01, actions=3, n_steps=10000, n_episodes=100,
                 run_type="Default", explore_temp=False, explore_lam=False,
                 explore_both=False, explore_weights=False, weights=.05,
                 greedy=False, verbose=False):

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
        self.min_lambda_ = 0

        # Choice of Random Action or Not
        self.greedy_flag = greedy
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
        self.number_of_actions = actions
        self.activity = {"Right": 0, "Left": 1, "Neutral": 2}
        self.action_index_ = {"1": 0, "-1": 1, "0": 2}
        self.last_action = None
        self.action = 0
        self.old_state = None
        self.state = [self.mountain_car.x, self.mountain_car.x_d]
        self.old_index = None
        self.index = self._get_index(self.state)
        # self.hold = np.zeros(3)

        # Trace Memory
        self.e = np.zeros((self.neuron_count, actions))
        if not explore_weights:
            self.weights = np.random.rand(self.neuron_count,
                                          actions)
            # self._normalize_weights()
        if explore_weights:
            self.weights = np.ones((self.neuron_count, actions)) * weights
            # self._normalize_weights()

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

        self.verbose = verbose

    def reset(self):
        self.mountain_car.reset()
        self.state = [self.mountain_car.x, self.mountain_car.x_d]
        self.e = np.zeros((self.neuron_count, self.number_of_actions))

    def initiate_trial(self, visual=False):
        # H5 Data Sets #
        h5data = dataSets.generate_data_sets(self.filename, self.centers)
        time_to_reward = [0]
        steps_to_reward = [0]
        # prepare for the visualization
        if visual:
            plb.ion()
            plb.pause(0.0001)
            mv = mountaincar.MountainCarViewer(self.mountain_car)
            mv.create_figure(self.n_steps, self.n_steps)
            plb.show()
        for episode_count in np.arange(self.n_episodes):
            self.reset()
            self._parameter_settings(episode_count)
            for step_count in range(self.n_steps):
                if self.verbose:
                    print("Episode: " + str(episode_count))
                    print("Simulation Step: {0}".format(str(step_count)))
                    print("Mountain Car state: {0}".format(str(self.state)))
                    print("Grid Center index: {0}".format(str(self.index)))
                    print("Grid Center: {0}".format(str(self.centers
                                                        [self.index])))
                self._learn()
                if visual:
                    # update the visualization
                    mv.update_figure()
                    plb.show()
                    plb.pause(0.0001)
                if self.mountain_car.R > 0.0:
                    print("Reward obtained at t = " + str(self.mountain_car.t))
                    steps_to_reward[0] = step_count
                    break
                elif step_count == self.n_steps - 1:
                    print("Maximum number of iterations reached.  No reward.")
                    steps_to_reward[0] = step_count
            time_to_reward[0] = self.mountain_car.t
            dataSets.generate_data_save(h5data, episode_count, time_to_reward,
                                        steps_to_reward, self.weights)

    def _parameter_settings(self, episode_count):
        if self.explore_temp:
            self.initial_temperature_ = self._time_decay(
                self.initial_temperature_, episode_count,
                self.min_temperature_)
        if self.explore_lam:
            self.lambda_ = self._time_decay(self.lambda_, episode_count,
                                            self.min_lambda_)
        if self.explore_both:
            self.initial_temperature_ = self._time_decay(
                self.initial_temperature_, episode_count,
                self.min_temperature_)
            self.lambda_ = self._time_decay(self.lambda_, episode_count,
                                            self.min_lambda_)

    def _learn(self):
        self._action_choice()

    def _input_layer(self, neuron_index, old_state_flag):
        if old_state_flag:
            rj = np.exp((-(neuron_index[0] - self.old_state[0]) ** 2) /
                        self.x_sigma - (
                            (neuron_index[1] - self.old_state[1]) ** 2) /
                        self.x_d_sigma)
        else:
            rj = np.exp((-(neuron_index[0] - self.state[0]) ** 2) /
                        self.x_sigma - (
                            (neuron_index[1] - self.state[1]) ** 2) /
                        self.x_d_sigma)
        return rj

    def _output_layer(self, action_index, old_state_flag):
        q_weights = 0.0
        for n in np.arange(self.neuron_count):
            q_weights += (self.weights[n][action_index] *
                          self._input_layer(self.centers[n], old_state_flag))
        # self.hold[action_index] = q_weights
        return q_weights

    def _update_eligibility(self):
        """
        Eligibility updates using the SARSA protocol.
        """
        index_action = self.action_index_["{}".format(self.action)]
        self.e *= self.lambda_ * self.gamma_
        self.e[self.old_index, index_action] += 1

    def _update_weights(self):
        """
        Weight updates using the SARSA protocol.
        """
        for i in range(self.neuron_count):
            for j in range(self.number_of_actions):
                self.weights[i][j] += self.eta_ * self.dirac * self.e[i][j]

    def _normalize_weights(self):
        un_weights = self.weights
        for i in np.arange(self.neuron_count):
            norm = np.sqrt(np.sum(np.square(un_weights[i])))
            self.weights[i] = un_weights[i] / norm

    def _td_error(self):
        reward = self.mountain_car.R
        index_last_action = self.action_index_["{}".format(self.last_action)]
        index_action = self.action_index_["{}".format(self.action)]
        self.dirac = (reward - (
            self._output_layer(index_last_action, True) -
            self.gamma_ * self._output_layer(index_action, False)))

    def _action_choice(self):
        """
        Choose the next action based on the current Q-values.
        Determined by soft-max rule.
        Cumulative probabilities for quick implementation.
        """

        if self.verbose:
            index = self._get_index(self.state)
            print("w(s, Right: {0}".format(str(self.weights[index][0])))
            print("w(s, Left: {0}".format(str(self.weights[index][1])))
            print("w(s, Neutral: {0}".format(str(self.weights[index][2])))
            print("Q(s, Right: " + str(self._output_layer(0, False)))
            print("Q(s, Left: {0}".format(str(self._output_layer(1, False))))
            print("Q(s, Neutral: " + str(self._output_layer(2, False)))

        prob_right = self._soft_max_rule(self.activity["Right"])
        prob_left = self._soft_max_rule(self.activity["Left"])
        prob_neutral = self._soft_max_rule(self.activity["Neutral"])

        c_prob_right = prob_right
        c_prob_left = c_prob_right + prob_left
        test_value = np.random.rand()

        if self.greedy_flag and (np.random.rand() <= self.initial_epsilon_):
            print("Greedy choice taken")
            if (prob_right > prob_left) and (prob_right > prob_neutral):
                c_prob_right = 1
                c_prob_left = 0
            elif (prob_left > prob_right) and (prob_left > prob_neutral):
                c_prob_right = 0
                c_prob_left = 1
            else:
                c_prob_right = 0
                c_prob_left = 0

        if self.verbose:
            print("Probability Right   : " + str(prob_right))
            print("Probability Left    : " + str(prob_left))
            print("Probability Neutral : " + str(prob_neutral))

        if test_value < c_prob_right:
            if self.verbose:
                print("Action taken: RIGHT!")
            self._update_state(1)
        elif test_value < c_prob_left:
            if self.verbose:
                print("Action taken: LEFT!")
            self._update_state(-1)
        else:
            if self.verbose:
                print("Action taken: neutral...")
            self._update_state(0)

    def _soft_max_rule(self, action_index):
        """
        Soft-max algorithm
        """
        if self.initial_temperature_ == np.inf:
            probability = 1 / 3
            return probability
        else:
            probability = (np.exp(self._output_layer(action_index, False)
                                  / self.initial_temperature_)
                           / self._all_actions())
            return probability

    def _all_actions(self):
        """
        For the denominator of the soft-max algorithm
        """
        total_q = 0.0
        for action_index in self.action_index_.values():
            total_q += np.exp(self._output_layer(action_index, False)
                              / self.initial_temperature_)
        return total_q

    def _update_state(self, command):
        self.last_action = self.action
        self.action = command
        self.old_x = self.mountain_car.x
        self.old_x_d = self.mountain_car.x_d
        self.old_state = [self.old_x, self.old_x_d]
        self.old_index = self.index
        self.mountain_car.apply_force(self.action)
        self.mountain_car.simulate_timesteps(self.time, self.dt)
        self.x = self.mountain_car.x
        self.x_d = self.mountain_car.x_d
        self.state = [self.x, self.x_d]
        self.index = self._get_index(self.state)
        self._td_error()
        self._update_eligibility()
        self._update_weights()

    def _get_index(self, state):
        return np.argmin(np.sum(np.square(np.abs(self.centers - state)), 1))

    def _time_decay(self, val, episode_count, min_val):
        return np.max(
            val * np.exp(-float(episode_count) / float(self.n_episodes)),
            min_val)


if __name__ == "__main__":
    t = 0
    while t < 10:
        d = DummyAgent(run_type="default", n_episodes=100, n_steps=10000,
                       neurons=20, eta=0.05, initial_temperature=1.0,
                       verbose=False,
                       explore_weights=True, weights=1.0)
        # greedy=True, initial_epsilon=0.1, verbose=True)
        d.initiate_trial(visual=False)
        t += 1

    # d = DummyAgent(explore_lam=True, run_type="explore_lam", n_episodes=100,
    #                n_steps=10000, neurons=20, eta=0.05 ** (20 ** 2))
    # d.initiate_trial()

    # d = DummyAgent(explore_temp=True, run_type="explore_temp",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(explore_both=True, run_type="explore_both",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(explore_weights=True, weights=0.0,\
    #                run_type="zero_weight",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(explore_weights=True, weights=1.0,
    # run_type="one_weight",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(initial_temperature=0.0001, run_type="zero_temp",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(initial_temperature=10e5, run_type="inf_temp",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
    #
    # d = DummyAgent(lam=0.0, run_type="zero_lambda",\
    #                n_episodes=100, n_steps=10000,\
    #                neurons=10, eta=0.05)
    # d.initiate_trial()
