from __future__ import division
import os
import numpy as np
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
from numba import jit

class Lattice(object):
    __slots__ = ("height",
                 "width",
                 "temperature",
                 "interaction_strength",
                 "beta",
                 "config",
                 "timestamp",
                 "plot_dir",
                 "path_fmt")

    def __init__(self,
                 height=64,
                 width=64,
                 temperature=0.4,
                 interaction_strength=2,
                 plot_dir="./plots"):
        """

        """
        assert temperature > 0

        # spin configuration
        self.config = np.random.choice([-1, 1], size=(height, width))
        self.beta = 1 / temperature
        self.timestamp = 0

        self.plot_dir = plot_dir
        self.path_fmt = os.path.join(self.plot_dir, "ising_model_sim-{:09d}.png")

        self.height = height
        self.width = width
        self.temperature = temperature
        self.interaction_strength = interaction_strength # J


    def single_spin_flip(self):
        """
        https://en.wikipedia.org/wiki/Ising_model
        """
        # STEP 1
        # Pick a spin site using selection probability g(mu, nu)
        # g(mu, nu) = 1 / N * N
        y = np.random.randint(low=0, high=self.height)
        x = np.random.randint(low=0, high=self.width)
        spin = self.config[y, x]

        # Calculate the contribution to the energy involving this spin.
        nearest_neighbors_spin = self.get_nearest_neighbors_spin(x=x, y=y)
        # energy = self.interaction_strength * spin * sum(nearest_neighbors_spin)

        # STEP 2
        # Flip the value of the spin and calculate the new contribution.
        # filpped_spin = -1 * spin
        # new_energy = -1 * J * filpped_spin * sum(nearest_neighbors_spin)
        #            = J * spin * sum(nearest_neighbors_spin)
        #            = -1 * energy

        # STEP 3
        # If the new energy is less, keep the flipped value.
        #  ==  If energy_diff is less than 0, keep the flipped value.
        #      energy_diff := new_energy - energy = 2 * new_energy
        energy_diff = 2 * self.interaction_strength * spin * sum(nearest_neighbors_spin)

        if energy_diff < 0:
            # acceptance_prob = 1
            self.config[y, x] *= -1 # keep the fillped value.
        else:
            # STEP 4
            # if the new energy is more, only keep with probability.
            acceptance_prob = np.exp(-1 * energy_diff * self.beta)
            if acceptance_prob > np.random.rand():
                self.config[y, x] *= -1 

        self.timestamp += 1

    def visualize(self, cmap="viridis"):
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(self.config, interpolation="none", cmap=cmap)

        energy = self.get_energy()
        magnetization = self.get_magnetization()

        plt.title("E: {} / M: {}".format(energy, magnetization))
        path = self.path_fmt.format(self.timestamp)
        fig.savefig(path)

    def get_nearest_neighbors_spin(self, x, y):
        nearest_neighbors_spin = [
            self.config[( y + 1) % self.height, x], # bottom
            self.config[( y - 1) % self.height, x], # top
            self.config[y, (x + 1) % self.width], # right
            self.config[y, (x - 1) % self.width] # left
        ]
        return nearest_neighbors_spin
 
    @jit
    def get_energy(self):
        energy = 0
        for h in range(self.height):
            for w in range(self.width):
                spin = self.config[h, w]
                nearest_neighbors = self.get_nearest_neighbors_spin(y=h, x=w)
                energy += -1 * self.interaction_strength * spin * sum(nearest_neighbors)
        # Consider overcounting
        energy = energy / 4
        return energy

    def get_magnetization(self):
        magnetization = self.config.mean()
        return magnetization



if __name__ == "__main__":
    lattice = Lattice(height=32, width=64)
    for timestamp in range(10000):
        lattice.single_spin_flip()
        if timestamp % 1000 == 0:
            lattice.visualize()
