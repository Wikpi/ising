import random as rng
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

n: int = 100

T_red: float = 10.0

k: int = int(1E5)

def main() -> None:
    rng.seed(0)

    configuration: list = generate_initial_configuration(n)

    fig, ax = plt.subplots()
    image = ax.imshow(configuration, cmap="gray", vmin=-1, vmax=1)

    for x in range(k + 1):
        configuration = MH(configuration, n)

        image.set_data(configuration)

        ax.set_title(f"{n}x{n} lattice computation = {x}") 
        
        plt.draw()
        # plt.pause(1/(n * k))

    save_figure(f"{n}x{n}-lattice-computation-T_red-{T_red}-runs-{k}", "data")
    plt.show()

    return

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

def update_configuration(configuration: list, index: int, n: int) -> list:
    # Create a new copy for the updated configuration.
    new_configuration: list = [x[:] for x in configuration]
    
    x, y = get_spin_coordinates(index, n)

    # Flip the spin of the selected spin in the new configuration.
    new_configuration[x][y] *= -1

    return new_configuration

def MH(configuration: list, n: int) -> list:
    i = rng.randint(0, n * n - 1)

    updated_configuration: list = update_configuration(configuration, i, n)

    delta_E = calculate_energy_change(updated_configuration, i, n)
    if delta_E <= 0:
        return updated_configuration
    
    R = rng.uniform(0, 1)
    p = get_boltzman_probability(delta_E)

    if R < p:
        return updated_configuration
    return configuration

def calculate_energy_change(configuration: list, i: int, n: int) -> float:
    # Get the value of the selected spin.
    spin_i = get_spin(configuration, i, n)

    # Get the value of the spins of the four nearest neighbors to the selected spin.
    spin_1 = get_spin(configuration, i - n, n)
    spin_2 = get_spin(configuration, i - 1, n)
    spin_3 = get_spin(configuration, i + 1, n)
    spin_4 = get_spin(configuration, i + n, n)

    # Calculate the energy change of the system due to flipping the selected spin.
    return -spin_i * (spin_1 + spin_2 + spin_3 + spin_4)

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

def get_spin(configuration: list, index: int, n) -> int:
    x, y = get_spin_coordinates(index, n)

    return configuration[x][y]

def get_boltzman_probability(E: float) -> float:
    return np.exp(-E/T_red)

def save_figure(filename: str = "simulation-result", output_dir: str = "data") -> None:
    # Ensure that the output directory is always present relative to the current working directory.
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "%s.png" % filename)

    plt.savefig(path)

main()