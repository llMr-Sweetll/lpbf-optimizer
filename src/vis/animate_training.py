
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def generate_melt_pool_animation(filename='melt_pool_dynamics.gif'):
    """
    Generates a physics-based animation of a moving laser melt pool.
    Simulates the Gaussian heat source interaction with the powder bed.
    """
    print(f"Generating animation: {filename}...")
    
    # Grid setup
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 2, 30)
    X, Y = np.meshgrid(x, y)
    
    # Material properties (Ti-6Al-4V approximation)
    P = 200 # Laser power
    v = 800 # Scan speed
    r0 = 0.2 # Beam radius
    
    # Setup figure
    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    plt.style.use('seaborn-v0_8-darkgrid')
    
    def update(frame):
        ax.clear()
        
        # Move laser source
        laser_x = 0.5 + (frame / 50) * 4.0
        laser_y = 1.0
        
        # Calculate temperature field (Analytic solution approx)
        # Rosenthal solution simplified
        r_squared = (X - laser_x)**2 + (Y - laser_y)**2
        T = 298 + (P / (2 * np.pi * r0)) * np.exp(-r_squared / (2 * r0**2)) 
        
        # Add thermal tail/lag
        T += 500 * np.exp(-(X - (laser_x - 0.5))**2 * 2 - (Y - laser_y)**2 * 5) * (X < laser_x)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, T, cmap='inferno', vmin=300, vmax=3000, alpha=0.9)
        
        # Plot laser position
        ax.scatter([laser_x], [laser_y], [3000], c='white', s=100, marker='*', edgecolors='red', label='Laser Source')
        
        # Styling
        ax.set_zlim(300, 3500)
        ax.set_xlabel('Scan Vector X (mm)')
        ax.set_ylabel('Width Y (mm)')
        ax.set_zlabel('Temperature (K)')
        ax.set_title(f"Melt Pool Thermal Dynamics | t = {frame*0.01:.2f} s | v = {v} mm/s")
        ax.view_init(elev=30, azim=-60 + frame * 0.5) # Rotate view
        
        # Add melting point plane
        # ax.plot_surface(X, Y, np.full_like(X, 1928), alpha=0.1, color='blue') # Ti64 Melting point
        
        return surf,

    ani = animation.FuncAnimation(fig, update, frames=60, interval=50, blit=False)
    
    # Save
    save_path = os.path.join('docs', filename)
    os.makedirs('docs', exist_ok=True)
    ani.save(save_path, writer='pillow', fps=20)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    generate_melt_pool_animation()
