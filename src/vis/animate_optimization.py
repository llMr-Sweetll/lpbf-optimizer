
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def generate_pareto_animation(filename='optimization_evolution.gif'):
    """
    Generates a 3D animation of the NSGA-III Optimization process.
    Visualizes the population evolving towards the Pareto Front.
    """
    print(f"Generating animation: {filename}...")

    # Simulation Parameters
    n_pop = 50
    n_gen = 60

    # Synthetic Pareto Front (ZDT1-like shape for demo)
    # Objectives: Minimize F1, F2, F3 (Stress, Porosity, Cost)

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    plt.style.use('dark_background')

    # Store history
    history = []

    for gen in range(n_gen):
        # Evolve population from random scatter to structured front
        progress = gen / n_gen

        # Random noise shrinking over time
        noise = 1.0 - progress

        # Underlying manifold
        u = np.linspace(0, 1, n_pop)
        v = np.linspace(0, 1, n_pop)
        U, V = np.meshgrid(u, v)
        U = U.flatten()
        V = V.flatten()

        # Objective Functions (synthetic)
        f1 = U
        f2 = V
        g = 1 + 9 * np.random.rand(len(U)) * noise # Convergence metric
        1 - np.sqrt(f1/g) - (f1/g) * np.sin(10*np.pi*f1)

        # Pareto approx
        x = f1 + np.random.normal(0, 0.1*noise, len(U))
        y = f2 + np.random.normal(0, 0.1*noise, len(U))
        z = (x**2 + y**2) * 0.5 + (1-progress)*2 # Desired front

        history.append((x[:100], y[:100], z[:100]))

    def update(frame):
        ax.clear()
        x, y, z = history[frame]

        # Scatter plot
        scat = ax.scatter(x, y, z, c=z, cmap='viridis', s=40 + frame/2, alpha=0.8, edgecolors='w', linewidth=0.5)

        # Labels and View
        ax.set_xlabel('Stress (MPa)')
        ax.set_ylabel('Porosity (%)')
        ax.set_zlabel('Build Time (h)')
        ax.set_title(f"NSGA-III Optimization Evolution | Gen: {frame}")
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.set_zlim(0, 3)

        # Rotating view
        ax.view_init(elev=30, azim=45 + frame * 0.5)

        return scat,

    ani = animation.FuncAnimation(fig, update, frames=n_gen, interval=50, blit=False)

    # Save
    save_path = os.path.join('docs', filename)
    os.makedirs('docs', exist_ok=True)
    ani.save(save_path, writer='pillow', fps=15)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    generate_pareto_animation()
