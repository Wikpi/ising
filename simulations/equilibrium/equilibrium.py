import random as rng
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

n: int = 50

T_red: float = 5.0

iterations: int = 1000


def main() -> None:
    rng.seed(0)

    configuration: list = generate_initial_configuration(n)

    m: np.ndarray = np.zeros(iterations)

    for i in range(iterations):
        for _ in range(n*n):
            configuration = MH(configuration, T_red)

        m[i] = abs(get_configuration_mean(configuration))

    plot_figure(m)


def get_configuration_mean(configuration: list) -> float:
    mean: float = 0

    n: int = len(configuration)

    for i in range(n):
        mean += sum(configuration[i])
    
    return mean / (n * n)


def generate_initial_configuration(n: int, random: bool = False) -> list:
    # Initialize the configuration of the system as a 2D-matrix of spins.
    new_configuration: list = []
    
    # Deterministic initial configuration of the system with all spins pointing in the same direction.
    spin = rng.choice([-1, 1])

    for x in range(n):
        new_configuration.append([])

        for y in range(n):
            # If random is set to True, then the initial configuration of the system will be randomized with spins pointing in random directions.
            if random:
                spin = rng.choice([-1, 1])
            new_configuration[x].append(spin)
    
    return new_configuration


def MH(configuration: list, T: float) -> list:
    n: int = len(configuration)
    
    i = rng.randint(0, n * n - 1)

    x, y = get_spin_coordinates(i, n)

    configuration[x][y] *= -1

    delta_E = calculate_energy_change(configuration, i, n)

    if delta_E <= 0:
        return configuration
    
    if rng.uniform(0, 1) < get_boltzman_probability(delta_E, T):
        return configuration

    configuration[x][y] *= -1
        
    return configuration


def calculate_energy_change(configuration: list, i: int, n: int) -> float:
    x, y = get_spin_coordinates(i, n)

    # Get the value of the spins of the four nearest neighbors to the selected spin.
    spin_up = configuration[(x - 1) % n][y]
    spin_down = configuration[(x + 1) % n][y]
    spin_left = configuration[x][(y - 1) % n]
    spin_right = configuration[x][(y + 1) % n]

    # Calculate the energy change of the system due to flipping the selected spin.
    return -2 * configuration[x][y] * (spin_up + spin_down + spin_left + spin_right)


def get_spin_coordinates(index: int, n: int):
    # The size of the 2D-matrix.
    N = n * n
    
    # Enforce periodic boundary conditions.
    if index >= N:
        index = index % N
    
    # Calculate the 2D-matrix row index for the given spin.
    x: int = int((index - index % n)/n)
    # Calculate the 2D-matrix column index for the given spin.
    y: int = int(index - x * n)

    return x, y


def get_boltzman_probability(E: float, T: float) -> float:
    return np.exp(-E/T)


def save_figure(filename: str = "simulation-result", output_dir: str = "data") -> None:
    # Ensure that the output directory is always present relative to the current working directory.
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "%s.png" % filename)

    plt.savefig(path)


def plot_figure(m: list) -> None:
    plt.plot(m, label="m")

    plt.legend()
    plt.xlabel("iteration") 
    plt.ylabel("|m|")
    plt.title("Ising model")

    save_figure(f"{n}x{n}-lattice-{T_red}_Tred-{iterations}_initial")

    plt.show()


if __name__ == "__main__":
    main()