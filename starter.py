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
                 temperature_half_life=1.0, neurons=5):

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

        self.neuron_count = neurons ** 2
        self.x_centers_distance = np.arange(-150, 30) / neurons
        self.phi_centers_distance = np.arange(-15, 15) / neurons
        self.centers = []
        for x in self.x_centers_distance:
            for phi in self.phi_centers_distance:
                self.centers.append([x, phi])

        self.activity = {"Forward": 0, "Reverse": 1, "Neutral": 2}

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

    def input_layer_activity(self, centers):
        rj = np.exp(((centers[0] - self.mountain_car.x) ** 2) /
                    self.x_centers_distance **
                    2 -
                    ((centers[1] - self.mountain_car.x_d) ** 2) /
                    self.phi_centers_distance
                    ** 2)
        return rj

    def output_layer_activity(self, command):
        action = self.activity["{}".format(command)]
        weights = self.output_layer_weights()
        q = 0.0
        for n in np.arange(self.neuron_count):
            q += weights[n][action] * self.input_layer_activity(
                self.centers[n])
        return q

    def output_layer_weights(self):
        w = np.ones((self.neuron_count, 3))
        # Code to update weights
        return w


if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
