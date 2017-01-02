import sys

import pylab as plb
import numpy as np
import mountaincar


class DummyAgent:
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, eta=0.1, gamma=.95, lam=0.0,
                 initial_epsilon=0.1, min_epsilon=0.0, half_life=1.0,
                 initial_temperature=1.0, min_temperature=0.01,
                 temperature_half_life=1.0, neurons=5, time=100,
                 dt=0.01):

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
        self.centers = [_x_space_, _x_d_space_]

        # Activity / State Parameters
        self.activity = {"Right": 0, "Left": 1, "Neutral": 2}
        self.action_index_ = {"1": 0, "-1": 1, "0": 2}
        self.last_action = None
        self.action = None
        self.old_state = None
        self.state = [self.mountain_car.x, self.mountain_car.x_d]

        # Trace Memory
        self.e = np.zeros(
            (self.neuron,
             self.neuron,
             len(self.action.values())))
        self.weights = 0.01 * np.random.rand(
            (self.neuron,
             self.neuron,
             len(self.action.values()))) + 0.1

        # Time step for Simulation
        self.time = time
        self.dt = dt

    def visualize_trial(self, n_steps=200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t),
            sys.stdout.flush()

            # choose a random action
            self.mountain_car.apply_force(-1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self):
        # This is your job!
        pass

    def input_layer_activity(self, neuron_index):
        rj = np.exp(((neuron_index[0] - self.state[0]) ** 2) /
                    self.x_centers_distance **
                    2 -
                    ((neuron_index[1] - self.state[1]) ** 2) /
                    self.phi_centers_distance
                    ** 2)
        return rj

    def output_layer_activity(self, action):
        weights = self.weights
        action_index = self.action_index_["{}".format(action)]
        q_weights = 0.0
        for n in np.arange(self.neuron_count):
            q_weights += weights[n][action_index] * self.input_layer_activity(
                self.centers[n])
        return q_weights

    def td_error(self):
        reward = self.mountain_car.R
        self.dirac = reward - (
            self.output_layer_activity(self.last_action) -
            self.gamma_ * self.output_layer_activity(
                self.action))

    def update_eligibility(self):
        """
        Eligibility updates using the SARSA protocol.
        """
        action = self.action_index_["{}".format(self.last_action)]
        self.e *= self.lambda_ * self.gamma_
        self.e[:, :, action] += 1

    def update_weights(self):
        """
        Weight updates using the SARSA protocol.
        """
        self.td_error()
        self.weights += self.dirac * self.eta_ * self.e

    def action_choice(self):
        """
        Choose the next action based on the current Q-values.
        Determined by soft-max rule.
        Cumulative probabilities for quick implementation.
        """

        c_prob_left = self.soft_max_rule(self.activity["Left"])
        c_prob_right = c_prob_left + self.soft_max_rule(self.activity["Right"])

        test_value = np.random.rand()

        if test_value < c_prob_left:
            self.update_state(-1)
        elif test_value < c_prob_right:
            self.update_state(1)
        else:
            self.update_state(0)

    def soft_max_rule(self, action):
        """
        Soft-max algorithm
        """
        probability = (np.exp(self.output_layer_activity(action)
                              / self.initial_temperature_)
                       / self.all_actions())
        return probability

    def all_actions(self):
        """
        For the denominator of the soft-max algorithm
        """
        total_q = 0.0
        for action in self.action_index_.values():
            total_q += np.exp(self.output_layer_activity(action))
        return total_q

    def update_state(self, command):
        self.last_action = self.action
        self.action = command
        self.old_x = self.mountain_car.x
        self.old_x_d = self.mountain_car.x_d
        self.old_state = [self.old_x, self.old_x_d]

        self.mountain_car.apply_force(self.action)
        self.mountain_car.simulate_timesteps(self.time, self.dt)

        self.x = self.mountain_car.x
        self.x_d = self.mountain_car.x_d
        self.state = [self.x, self.x_d]
        self.update_eligibility()


if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
