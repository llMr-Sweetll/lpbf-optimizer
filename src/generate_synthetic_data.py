import argparse
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from data.lineage import write_manifest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('synthetic_data')


class SyntheticDataGenerator:
    """
    Generate synthetic data that mimics FEA simulation results for LPBF process.

    This class creates realistic synthetic data for training and testing the PINN model
    when actual FEA simulation data is not available.
    """

    def __init__(self, config_path):
        """
        Initialize the synthetic data generator

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up paths
        self.output_dir = Path(self.config['data']['processed_data_path']).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get material properties for physics-based data generation
        self.mat_props = self.config['material_properties']

        # Path to save the synthetic dataset
        self.output_path = Path(self.config['data']['processed_data_path'])

    def generate_scan_vectors(self, n_samples):
        """
        Generate scan vectors (process parameters)

        Args:
            n_samples: Number of sample points to generate

        Returns:
            Array of scan vectors
        """
        # Get parameter bounds from config
        param_bounds = self.config['optimizer']['param_bounds']

        # Generate random values within bounds for each parameter
        scan_vectors = {}
        for param, bounds in param_bounds.items():
            scan_vectors[param] = np.random.uniform(bounds[0], bounds[1], n_samples)

        # Stack into a single array
        param_names = list(param_bounds.keys())
        scan_array = np.column_stack([scan_vectors[param] for param in param_names])

        return scan_array, param_names

    def generate_coordinates(self, n_samples, n_points_per_sample):
        """
        Generate spatial coordinates

        Args:
            n_samples: Number of scan vector samples
            n_points_per_sample: Number of points to evaluate per scan vector

        Returns:
            Tuple of (coords, times)
        """
        # For each scan vector, generate multiple spatial points
        # Create a small "coupon" for each scan vector with points in a grid

        # Define coupon size (in mm)
        x_range = (-5, 5)
        y_range = (-5, 5)
        z_range = (0, 1)

        # Generate grid points for each sample
        all_coords = []
        all_times = []

        for _ in range(n_samples):
            # Create a grid for this sample
            points_per_dim = int(np.cbrt(n_points_per_sample))
            x = np.linspace(x_range[0], x_range[1], points_per_dim)
            y = np.linspace(y_range[0], y_range[1], points_per_dim)
            z = np.linspace(z_range[0], z_range[1], points_per_dim)

            # Create meshgrid
            X, Y, Z = np.meshgrid(x, y, z)

            # Reshape to points
            coords = np.column_stack([
                X.flatten(),
                Y.flatten(),
                Z.flatten()
            ])

            # Create time points (all at final state for simplicity)
            # In a more complex simulation, we might vary time
            times = np.ones((len(coords), 1))

            all_coords.append(coords)
            all_times.append(times)

        # Stack all coordinates
        coords = np.vstack(all_coords)
        times = np.vstack(all_times)

        return coords, times

    def generate_fea_outputs(self, scan_vectors, coords, times):
        """
        Generate simulated outputs based on physical relationships

        Args:
            scan_vectors: Array of process parameters
            coords: Spatial coordinates
            times: Time points

        Returns:
            Array of outputs [residual_stress, porosity, geometric_accuracy]
        """
        # Extract scan vector parameters
        n_params = scan_vectors.shape[1]

        if n_params >= 1:
            P = scan_vectors[:, 0]  # Laser power (W)
        else:
            P = np.ones(len(scan_vectors)) * 200

        if n_params >= 2:
            v = scan_vectors[:, 1]  # Scan speed (mm/s)
        else:
            v = np.ones(len(scan_vectors)) * 800

        if n_params >= 3:
            h = scan_vectors[:, 2]  # Hatch spacing (mm)
        else:
            h = np.ones(len(scan_vectors)) * 0.1

        if n_params >= 4:
            theta = scan_vectors[:, 3]  # Scan angle (degrees)
        else:
            theta = np.ones(len(scan_vectors)) * 45

        # Repeat scan vectors to match coordinates
        points_per_sample = len(coords) // len(scan_vectors)
        P = np.repeat(P, points_per_sample)
        v = np.repeat(v, points_per_sample)
        h = np.repeat(h, points_per_sample)
        theta = np.repeat(theta, points_per_sample)

        # Use the last parameter column as layer thickness (consistent with config).
        layer_thickness = scan_vectors[:, -1]
        layer_thickness = np.repeat(layer_thickness, points_per_sample)

        # Calculate volumetric energy density (J/mm³)
        energy_density = P / (v * h * layer_thickness)

        # Calculate cooling rate (simplified)
        # In reality, this would be based on heat equation solution
        cooling_rate = P / (v * np.sqrt(h))

        # Add spatial variation (e.g., higher temperatures at center)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # Distance from center affects properties
        r = np.sqrt(x**2 + y**2)
        spatial_factor = np.exp(-r**2 / 25)  # Gaussian decay from center

        # Height effect (properties change with z)
        z_factor = np.sin(np.pi * z) + 0.5

        # 1. Residual Stress (MPa)
        # Higher power, lower speed, and lower hatch spacing increase stress
        base_stress = 200 + 0.5*P - 0.1*v + 200*h
        # Add cooling rate effect (faster cooling -> higher stress)
        cooling_effect = 0.05 * cooling_rate
        # Add spatial variation
        spatial_stress = 50 * spatial_factor + 20 * z_factor
        # Add noise
        stress_noise = np.random.normal(0, 25, len(P))

        residual_stress = base_stress + cooling_effect + spatial_stress + stress_noise
        residual_stress = np.clip(residual_stress, 50, 800)  # Reasonable range for LPBF

        # 2. Porosity (%)
        # Energy density too low or too high increases porosity (U-shaped curve)
        optimal_ed = 25.0  # J/mm³ (representative middle of the realistic range)
        base_porosity = 0.05 + 0.2 * (1.0 / (1.0 + np.exp(-(energy_density - optimal_ed) / 10.0)))
        # Add cooling rate effect (too fast or too slow can increase porosity)
        cooling_porosity = 0.02 * np.abs(cooling_rate - 100) / 100
        # Add spatial variation
        spatial_porosity = 0.01 * (1 - spatial_factor) + 0.01 * z_factor
        # Add noise
        porosity_noise = np.abs(np.random.normal(0, 0.005, len(P)))

        porosity = base_porosity + cooling_porosity + spatial_porosity + porosity_noise
        porosity = np.clip(porosity, 0.001, 0.3)  # 0.1% to 30% is a reasonable synthetic range

        # 3. Geometric Accuracy Ratio (dimensionless)
        # Energy density affects accuracy (too high or too low reduces accuracy)
        base_accuracy = 0.9 + 0.1 * np.exp(-(energy_density - optimal_ed)**2 / 200.0)
        # Add scan angle effect (certain angles may have better accuracy)
        angle_effect = 0.02 * np.sin(theta * np.pi / 180)
        # Add spatial variation
        spatial_accuracy = 0.05 * spatial_factor - 0.03 * z_factor
        # Add noise
        accuracy_noise = np.random.normal(0, 0.02, len(P))

        geometric_accuracy = base_accuracy + angle_effect + spatial_accuracy + accuracy_noise
        geometric_accuracy = np.clip(geometric_accuracy, 0.7, 1.0)  # 70% to 100% accuracy

        # Combine outputs
        outputs = np.column_stack([residual_stress, porosity, geometric_accuracy])

        return outputs

    def split_data(self, scan_ids, n_scan_vectors):
        """
        Split scan-vector IDs into training, validation, and test sets.

        The split is performed at the scan-vector level so that all spatial
        points belonging to the same vector stay in the same split, preventing
        data leakage.

        Args:
            scan_ids (np.ndarray): Per-point scan vector IDs.
            n_scan_vectors (int): Total number of scan vectors.

        Returns:
            Tuple of (train_indices, val_indices, test_indices) scan-vector IDs.
        """
        # Get split ratios from config
        train_split = self.config['data'].get('train_split', 0.8)
        val_split = self.config['data'].get('val_split', 0.1)

        # Create scan-vector indices and shuffle
        indices = np.random.permutation(n_scan_vectors)

        # Calculate split points
        n_train = int(train_split * n_scan_vectors)
        n_val = int(val_split * n_scan_vectors)
        n_test = n_scan_vectors - n_train - n_val

        # Ensure every split receives at least one scan vector when possible
        if n_val == 0 and n_train > 1:
            n_train -= 1
            n_val += 1
        if n_test == 0:
            if n_val > 1:
                n_val -= 1
                n_test += 1
            elif n_train > 1:
                n_train -= 1
                n_test += 1

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return train_indices, val_indices, test_indices

    def generate(self, n_scan_vectors=100, n_points_per_vector=1000):
        """
        Generate a complete synthetic dataset

        Args:
            n_scan_vectors: Number of different scan vector combinations
            n_points_per_vector: Number of spatial points per scan vector

        Returns:
            Path to the generated dataset
        """
        logger.info(f"Generating synthetic dataset with {n_scan_vectors} scan vectors and {n_points_per_vector} points per vector")

        # Generate scan vectors
        scan_vectors, param_names = self.generate_scan_vectors(n_scan_vectors)
        logger.info(f"Generated scan vectors with parameters: {param_names}")

        # Generate spatial coordinates
        coords, times = self.generate_coordinates(n_scan_vectors, n_points_per_vector)
        logger.info(f"Generated {len(coords)} coordinate points")

        # Calculate actual points per vector (since grid generation might round down)
        actual_points_per_vector = len(coords) // n_scan_vectors

        # Generate outputs based on physics-informed relations
        outputs = self.generate_fea_outputs(scan_vectors, coords, times)
        logger.info(f"Generated outputs with shape {outputs.shape}")

        # Combine inputs for the neural network
        # Each input is [scan_vector, coordinates, time]
        scan_vectors_expanded = np.repeat(scan_vectors, actual_points_per_vector, axis=0)
        inputs = np.hstack([scan_vectors_expanded, coords, times])

        # Per-point scan-vector IDs for leakage-free splitting
        scan_ids = np.repeat(np.arange(n_scan_vectors), actual_points_per_vector)

        # Compute standard-scaler parameters from the full generated arrays
        scan_mean = scan_vectors.mean(axis=0)
        scan_std = scan_vectors.std(axis=0)
        coords_mean = coords.mean(axis=0)
        coords_std = coords.std(axis=0)

        # Split data by scan vector to prevent leakage
        train_vec_ids, val_vec_ids, test_vec_ids = self.split_data(scan_ids, n_scan_vectors)

        train_mask = np.isin(scan_ids, train_vec_ids)
        val_mask = np.isin(scan_ids, val_vec_ids)
        test_mask = np.isin(scan_ids, test_vec_ids)

        train_data = {
            'inputs': inputs[train_mask],
            'outputs': outputs[train_mask],
            'coordinates': coords[train_mask],
            'time': times[train_mask],
            'scan_vectors': scan_vectors_expanded[train_mask]
        }
        val_data = {
            'inputs': inputs[val_mask],
            'outputs': outputs[val_mask],
            'coordinates': coords[val_mask],
            'time': times[val_mask],
            'scan_vectors': scan_vectors_expanded[val_mask]
        }
        test_data = {
            'inputs': inputs[test_mask],
            'outputs': outputs[test_mask],
            'coordinates': coords[test_mask],
            'time': times[test_mask],
            'scan_vectors': scan_vectors_expanded[test_mask]
        }
        logger.info(
            f"Split data into {len(train_data['inputs'])} training, "
            f"{len(val_data['inputs'])} validation, and {len(test_data['inputs'])} test samples"
        )

        # Save to HDF5 file
        with h5py.File(self.output_path, 'w') as f:
            # Create groups
            train_group = f.create_group('train')
            val_group = f.create_group('val')
            test_group = f.create_group('test')
            meta_group = f.create_group('metadata')

            # Save split data
            for group, data in [(train_group, train_data),
                                (val_group, val_data),
                                (test_group, test_data)]:
                group.create_dataset('inputs', data=data['inputs'])
                group.create_dataset('outputs', data=data['outputs'])
                group.create_dataset('coordinates', data=data['coordinates'])
                group.create_dataset('time', data=data['time'])
                group.create_dataset('scan_vectors', data=data['scan_vectors'])

            # Save metadata
            meta_group.attrs['n_total'] = len(inputs)
            meta_group.attrs['n_train'] = len(train_data['inputs'])
            meta_group.attrs['n_val'] = len(val_data['inputs'])
            meta_group.attrs['n_test'] = len(test_data['inputs'])
            meta_group.attrs['created'] = datetime.now().isoformat()

            # Save scan-vector IDs for each split
            meta_group.create_dataset('train_scan_ids', data=train_vec_ids)
            meta_group.create_dataset('val_scan_ids', data=val_vec_ids)
            meta_group.create_dataset('test_scan_ids', data=test_vec_ids)

            # Save standard-scaler parameters
            scaler_group = meta_group.create_group('scaler_params')
            sv_scaler = scaler_group.create_group('scan_vectors')
            sv_scaler.create_dataset('mean', data=scan_mean)
            sv_scaler.create_dataset('std', data=scan_std)
            coord_scaler = scaler_group.create_group('coordinates')
            coord_scaler.create_dataset('mean', data=coords_mean)
            coord_scaler.create_dataset('std', data=coords_std)

            # Save parameter names
            dt = h5py.special_dtype(vlen=str)
            param_names_array = np.array(param_names, dtype=object)
            meta_group.create_dataset('parameter_names', data=param_names_array, dtype=dt)

        logger.info(f"Saved synthetic dataset to {self.output_path}")

        # Write a lightweight lineage manifest next to the dataset
        manifest_path = write_manifest(self.config, self.output_path, self.output_dir)
        logger.info(f"Saved dataset manifest to {manifest_path}")

        return self.output_path


def main():
    """Main function to generate synthetic data"""
    parser = argparse.ArgumentParser(description='Generate synthetic FEA data for LPBF optimization')
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--scan-vectors', type=int, default=100,
                        help='Number of scan vector combinations to generate')
    parser.add_argument('--points-per-vector', type=int, default=1000,
                        help='Number of spatial points per scan vector')

    args = parser.parse_args()

    # Create generator
    generator = SyntheticDataGenerator(args.config)

    # Generate dataset
    output_path = generator.generate(args.scan_vectors, args.points_per_vector)

    print(f"Synthetic data generation complete. Dataset saved to {output_path}")


if __name__ == '__main__':
    main()
