import random as rng
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

# Size of an individual square lattice.
n: int = 50
# Sample size.
k: int = 10

equilibrium_iterations: int = 500
measure_iterations: int = 100

# T_red domain.
T_min: float = 0.1
T_max: float = 273
T_n: int = 50

J_min: float = 1e-22
J_max: float = 1e-21
J_n: int = 50


def main() -> None:
    rng.seed(0)

    T: np.ndarray = np.linspace(T_min, T_max, T_n)
    J: np.ndarray = np.linspace(J_min, J_max, J_n)
    
    T_red: np.ndarray = sp.constants.k * T / J

    m_sim: np.ndarray = np.zeros_like(T_red)

    for i in range(len(T_red)):
        configuration: list = generate_initial_configuration(n)

        configuration = MH(configuration, equilibrium_iterations, T_red[i])

        m_sim[i] = get_average_magnetization(configuration, k, T_red[i])
    
    T_c_theory: float = 2 / np.log(1 + np.sqrt(2))
    T_c_sim: float = 0#approximate_critical_temperature()

    m_theory: np.ndarray = np.zeros_like(T_red)
    m_theory[T_red < T_c_theory] = (1 - np.sinh(2 / T_red[T_red < T_c_theory]) ** (-4)) ** (1/8)

    plot_figure(
        [
            [T_red, m_sim],
            [T_red, m_theory],
        ],
        [
            T_c_sim, 
            T_c_theory,
        ],
    )


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

    for i in range(n):
        new_configuration.append([])

        for _ in range(n):
            # If random is set to True, then the initial configuration of the system will be randomized with spins pointing in random directions.
            if random:
                spin = rng.choice([-1, 1])
            new_configuration[i].append(spin)
    
    return new_configuration


def MH(configuration: list, iterations: int, T: float) -> list:
    n: int = len(configuration)
    
    for _ in range(iterations * n ** 2):
        i = rng.randint(0, n * n - 1)

        x, y = get_spin_coordinates(i, n)

        configuration[x][y] *= -1

        delta_E = calculate_energy_change(configuration, i, n)

        if delta_E <= 0:
            continue
        
        if rng.uniform(0, 1) < get_boltzman_probability(delta_E, T):
            continue

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
    index = index % N
    
    # Calculate the 2D-matrix row index for the given spin.
    x: int = int((index - index % n)/n)
    # Calculate the 2D-matrix column index for the given spin.
    y: int = int(index - x * n)

    return x, y


def get_boltzman_probability(E: float, T: float) -> float:
    return np.exp(-E/T)


def get_average_magnetization(configuration: list, samples: int, T) -> float:
    n: int = len(configuration)
    
    m_average: float = 0

    for _ in range(samples):
        configuration = MH(configuration, measure_iterations, T)

        m_average += abs(get_configuration_mean(configuration))
        
    return m_average / samples


def approximate_critical_temperature() -> float:
    dm_dT = np.diff(m_average) / np.diff(T_red)

    index = np.argmin(dm_dT)
    T_c = (T_red[index] + T_red[index + 1]) / 2

    print(f"Estimated Tc = {T_c}")

    return T_c


def plot_figure(m_list: list, T_c_list: list) -> None:
    m_sim_points, m_theory_points = m_list

    plt.plot(m_sim_points[0], m_sim_points[1], label="Simulation", color="blue")
    plt.plot(m_theory_points[0], m_theory_points[1], label="Theoretical", color="orange")

    T_c_sim, T_c_theory = T_c_list

    # plt.axvline(T_c_sim, linestyle="-.", label="Extimated $T_c$", color="purple")
    plt.axvline(T_c_theory, linestyle="--", label="Theoretical $T_c$", color="red")

    plt.legend()
    plt.xlabel("$T_{red}$") 
    plt.ylabel("|m|")
    plt.title("Ising model")

    save_figure(f"{n}x{n}-lattice-{k}_samples-{equilibrium_iterations}_equilibrium-{measure_iterations}_measure")

    plt.show()


def save_figure(filename: str = "simulation-result", output_dir: str = "data") -> None:
    # Ensure that the output directory is always present relative to the current working directory.
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "%s.png" % filename)

    plt.savefig(path)


if __name__ == "__main__":
    main()