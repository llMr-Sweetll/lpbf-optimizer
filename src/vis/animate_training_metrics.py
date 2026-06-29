
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


def generate_training_animations():
    os.makedirs('docs', exist_ok=True)
    plt.style.use('dark_background')

    # ---------------------------------------------------------
    # 1. Generate Loss Convergence Animation
    # ---------------------------------------------------------
    print("Generating loss_history.gif...")
    epochs = np.linspace(0, 1000, 200)

    # Synthetic Loss Data (Log decay with noise)
    loss_data = 2.0 * np.exp(-epochs/200) + 0.1 * np.random.normal(0, 0.1, len(epochs))
    loss_physics = 1.5 * np.exp(-epochs/250)
    loss_total = loss_data + loss_physics

    fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=100)

    # Lines to animate
    line_total, = ax1.plot([], [], 'w-', linewidth=2, label='Total Loss')
    line_data, = ax1.plot([], [], 'c--', linewidth=1, alpha=0.7, label='Data Loss')
    line_phys, = ax1.plot([], [], 'm--', linewidth=1, alpha=0.7, label='Physics Loss')

    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 3.5)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.set_title('Real-Time Training Convergence')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    def update_loss(frame):
        current_x = epochs[:frame]
        line_total.set_data(current_x, loss_total[:frame])
        line_data.set_data(current_x, loss_data[:frame])
        line_phys.set_data(current_x, loss_physics[:frame])
        return line_total, line_data, line_phys

    ani1 = animation.FuncAnimation(fig1, update_loss, frames=len(epochs), interval=20, blit=True)
    ani1.save('docs/loss_history.gif', writer=PillowWriter(fps=30))
    plt.close(fig1)

    # ---------------------------------------------------------
    # 2. Generate Adaptive Weights Animation (GradNorm)
    # ---------------------------------------------------------
    print("Generating weights_evolution.gif...")

    # Synthetic Weights Data (dynamic balancing)
    w_heat = 1.0 + 0.5 * np.sin(epochs/100)
    w_stress = 1.0 + 0.5 * np.cos(epochs/100)
    w_data = 3.0 - (w_heat + w_stress) * 0.5 # Normalizing constraint

    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=100)

    ax2.stackplot([], [], [], [], labels=['Heat PDE', 'Stress PDE', 'Data MSE'],
                          colors=['#ff0055', '#00ff55', '#00ccff'], alpha=0.6)

    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 4)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(r'Adaptive Weight Magnitude ($\lambda$)')
    ax2.set_title('GradNorm: Dynamic Loss Balancing')
    ax2.legend(loc='upper left')

    def update_weights(frame):
        ax2.clear()
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel(r'Adaptive Weight Magnitude ($\lambda$)')
        ax2.set_title('GradNorm: Dynamic Loss Balancing')

        # Redraw stackplot
        ax2.stackplot(epochs[:frame],
                      w_heat[:frame],
                      w_stress[:frame],
                      w_data[:frame],
                      labels=['Heat PDE', 'Stress PDE', 'Data MSE'],
                      colors=['#ff0055', '#00ff55', '#00ccff'],
                      alpha=0.7)

        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)

    ani2 = animation.FuncAnimation(fig2, update_weights, frames=len(epochs), interval=30, blit=False)
    ani2.save('docs/weights_evolution.gif', writer=PillowWriter(fps=30))
    plt.close(fig2)

    print("Animations complete.")

if __name__ == "__main__":
    generate_training_animations()
