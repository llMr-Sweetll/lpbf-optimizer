import argparse
import logging
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from skimage import measure

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('characterise')


def parse_xct_porosity_csv(path):
    """Parse an XCT porosity CSV file.

    The CSV must contain a ``porosity`` column. Optional ``x``, ``y`` and ``z``
    spatial columns are preserved when present.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (DataFrame, summary_dict) where summary_dict contains the
        mean, standard deviation, minimum and maximum porosity values.

    Raises:
        ValueError: If the required ``porosity`` column is missing.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if 'porosity' not in df.columns:
        raise ValueError(
            f"XCT porosity CSV '{path}' is missing required column 'porosity'. "
            f"Found columns: {list(df.columns)}"
        )

    summary = {
        'mean': float(df['porosity'].mean()),
        'std': float(df['porosity'].std()),
        'min': float(df['porosity'].min()),
        'max': float(df['porosity'].max()),
    }
    return df, summary


def parse_ebsd_csv(path):
    """Parse an EBSD CSV file.

    The CSV must contain ``phi1``, ``phi``, ``phi2``, ``x`` and ``y`` columns.
    Optional ``ci`` (confidence index) and ``phase`` columns are preserved.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the EBSD data.

    Raises:
        ValueError: If any required column is missing.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = ['phi1', 'phi', 'phi2', 'x', 'y']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"EBSD CSV '{path}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def parse_stress_csv(path):
    """Parse a residual stress CSV file.

    The CSV must contain a ``sigma_xx`` column. Optional ``sigma_yy``,
    ``sigma_zz`` and spatial ``x``, ``y``, ``z`` columns are used when present.
    The von Mises equivalent stress is computed from the available normal
    components.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (DataFrame, summary_dict). The DataFrame includes an added
        ``von_mises`` column. ``summary_dict`` contains mean, max and standard
        deviation of the von Mises stress plus mean values for each available
        normal component.

    Raises:
        ValueError: If the required ``sigma_xx`` column is missing.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if 'sigma_xx' not in df.columns:
        raise ValueError(
            f"Stress CSV '{path}' is missing required column 'sigma_xx'. "
            f"Found columns: {list(df.columns)}"
        )

    sigma_xx = df['sigma_xx'].values
    sigma_yy = df['sigma_yy'].values if 'sigma_yy' in df.columns else np.zeros_like(sigma_xx)
    sigma_zz = df['sigma_zz'].values if 'sigma_zz' in df.columns else np.zeros_like(sigma_xx)

    von_mises = np.sqrt(
        0.5 * (
            (sigma_xx - sigma_yy) ** 2
            + (sigma_yy - sigma_zz) ** 2
            + (sigma_zz - sigma_xx) ** 2
        )
    )
    df = df.copy()
    df['von_mises'] = von_mises

    summary = {
        'mean_von_mises': float(np.mean(von_mises)),
        'max_von_mises': float(np.max(von_mises)),
        'std_von_mises': float(np.std(von_mises)),
        'mean_xx': float(np.mean(sigma_xx)),
        'mean_yy': float(np.mean(sigma_yy)),
        'mean_zz': float(np.mean(sigma_zz)),
    }
    return df, summary


class LPBFCharacterisation:
    """
    Process and analyze experimental data from LPBF-built samples, including
    X-ray CT (XCT) and electron backscatter diffraction (EBSD) data.

    This class helps validate the simulation and optimization results by
    comparing predicted properties with actual measured properties.
    """

    def __init__(self, config_path):
        """
        Initialize the characterization system

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up paths
        self.output_dir = Path(self.config.get('validate', {}).get('characterisation_dir', 'characterisation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        # Set up log file for this run
        file_handler = logging.FileHandler(self.run_dir / 'characterisation.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Save config for this run
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

    def load_xct_data(self, xct_file):
        """
        Load X-ray CT data from file

        Args:
            xct_file: Path to XCT data file (TIFF stack, HDF5, or other format)

        Returns:
            3D numpy array of CT data
        """
        file_ext = Path(xct_file).suffix.lower()

        if file_ext in ['.tif', '.tiff']:
            # Load TIFF stack
            try:
                from skimage.io import imread
                data = imread(xct_file)
                if data.ndim == 3:
                    logger.info(f"Loaded TIFF stack with shape {data.shape}")
                    return data
                else:
                    logger.error(f"Expected 3D data, got {data.ndim}D data")
                    return None
            except Exception as e:
                logger.error(f"Failed to load TIFF stack: {e}")
                return None

        elif file_ext == '.h5':
            # Load HDF5
            try:
                with h5py.File(xct_file, 'r') as f:
                    # Assuming the main dataset is called 'volume'
                    # This might need to be adjusted based on the specific HDF5 format
                    if 'volume' in f:
                        data = f['volume'][:]
                        logger.info(f"Loaded HDF5 volume with shape {data.shape}")
                        return data
                    else:
                        available_keys = list(f.keys())
                        logger.warning(f"'volume' not found in HDF5 file. Available keys: {available_keys}")

                        # Try to use the first dataset that looks like volume data
                        for key in available_keys:
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                                data = f[key][:]
                                logger.info(f"Using dataset '{key}' with shape {data.shape}")
                                return data

                        logger.error("No suitable 3D dataset found in HDF5 file")
                        return None
            except Exception as e:
                logger.error(f"Failed to load HDF5 file: {e}")
                return None

        elif file_ext == '.raw':
            # Load raw binary data - would need additional parameters for dimensions
            # For demonstration, we'll create a small random volume
            logger.warning("RAW format loading needs additional parameters. Using random data for demonstration.")
            return np.random.rand(100, 100, 100)

        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return None

    def analyze_porosity(self, xct_data, threshold=None):
        """
        Analyze porosity from XCT data

        Args:
            xct_data: 3D numpy array of CT data
            threshold: Threshold value for segmentation (if None, use Otsu's method)

        Returns:
            Dictionary with porosity metrics
        """
        from scipy import ndimage
        from skimage import measure
        from skimage.filters import threshold_otsu

        # If no threshold provided, use Otsu's method
        if threshold is None:
            threshold = threshold_otsu(xct_data)

        # Segment the volume
        binary_volume = xct_data < threshold  # Assuming pores are darker

        # Calculate overall porosity
        total_volume = binary_volume.size
        pore_volume = np.sum(binary_volume)
        porosity = pore_volume / total_volume

        # Label connected components (pores)
        labeled_pores, num_pores = ndimage.label(binary_volume)

        # Calculate properties of each pore
        props = measure.regionprops(labeled_pores)

        # Extract pore sizes and calculate distribution
        pore_sizes = [prop.area for prop in props]

        # Calculate average pore size
        avg_pore_size = np.mean(pore_sizes) if pore_sizes else 0

        # Calculate largest pore size
        max_pore_size = np.max(pore_sizes) if pore_sizes else 0

        # Calculate pore size distribution
        if pore_sizes:
            hist, bin_edges = np.histogram(pore_sizes, bins=20)
            pore_distribution = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            pore_distribution = {
                'histogram': [],
                'bin_edges': []
            }

        # Save visualization
        # Take middle slice for visualization
        middle_slice = binary_volume.shape[0] // 2
        slice_image = binary_volume[middle_slice, :, :]

        plt.figure(figsize=(10, 8))
        plt.imshow(slice_image, cmap='gray')
        plt.title(f"Porosity Slice (z={middle_slice})")
        plt.colorbar(label='Pore')
        plt.savefig(self.run_dir / 'porosity_slice.png')
        plt.close()

        # Results
        results = {
            'porosity': porosity,
            'num_pores': num_pores,
            'avg_pore_size': avg_pore_size,
            'max_pore_size': max_pore_size,
            'pore_distribution': pore_distribution,
            'threshold': threshold
        }

        # Save results
        with open(self.run_dir / 'porosity_results.yaml', 'w') as f:
            yaml.dump(results, f)

        logger.info(f"Porosity analysis completed: {porosity:.2%} porosity with {num_pores} pores")
        return results

    def load_ebsd_data(self, ebsd_file):
        """
        Load EBSD (Electron Backscatter Diffraction) data from file

        Args:
            ebsd_file: Path to EBSD data file (usually .ang, .ctf or .csv)

        Returns:
            Dictionary with EBSD data

        Raises:
            ValueError: If the file format is unsupported or parsing fails.
        """
        file_ext = Path(ebsd_file).suffix.lower()

        if file_ext == '.ang':
            ebsd_data = self._parse_ang_file(ebsd_file)
        elif file_ext == '.ctf':
            ebsd_data = self._parse_ctf_file(ebsd_file)
        elif file_ext == '.csv':
            df = parse_ebsd_csv(ebsd_file)
            ebsd_data = {col: df[col].values for col in df.columns}
        else:
            raise ValueError(
                f"Unsupported EBSD file format '{file_ext}' for '{ebsd_file}'. "
                "Supported formats are .ang, .ctf and .csv."
            )

        if ebsd_data is None:
            raise RuntimeError(f"Failed to load EBSD data from '{ebsd_file}'.")

        return ebsd_data

    def _parse_ang_file(self, ang_file):
        """
        Parse TSL .ang EBSD file format

        Args:
            ang_file: Path to .ang file

        Returns:
            Dictionary with EBSD data

        Raises:
            ValueError: If the file does not appear to be in TSL .ang format.
            RuntimeError: If parsing fails for any other reason.
        """
        try:
            # Read header to determine format
            header_lines = 0
            with open(ang_file, 'r') as f:
                line = f.readline()
                while line.startswith('#'):
                    header_lines += 1
                    line = f.readline()

            # Read data with pandas
            data = pd.read_csv(ang_file, skiprows=header_lines, sep=r'\s+', header=None)

            # TSL .ang format typically has columns:
            # phi1, phi, phi2, x, y, IQ, CI, Phase, ...
            if data.shape[1] < 8:
                raise ValueError(
                    f"File '{ang_file}' does not appear to be in TSL .ang format: "
                    f"expected at least 8 columns, found {data.shape[1]}."
                )

            ebsd_data = {
                'phi1': data.iloc[:, 0].values,
                'phi': data.iloc[:, 1].values,
                'phi2': data.iloc[:, 2].values,
                'x': data.iloc[:, 3].values,
                'y': data.iloc[:, 4].values,
                'iq': data.iloc[:, 5].values,
                'ci': data.iloc[:, 6].values,
                'phase': data.iloc[:, 7].values.astype(int)
            }

            # Try to determine grid dimensions
            x_unique = len(np.unique(ebsd_data['x']))
            y_unique = len(np.unique(ebsd_data['y']))

            # Reshape data if it's on a regular grid
            if x_unique * y_unique == len(ebsd_data['x']):
                for key in ebsd_data:
                    ebsd_data[key] = ebsd_data[key].reshape(y_unique, x_unique)

            logger.info(f"Loaded EBSD data from {ang_file} with {len(ebsd_data['x'])} points")
            return ebsd_data
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to parse .ang file '{ang_file}': {e}") from e

    def _parse_ctf_file(self, ctf_file):
        """
        Parse HKL .ctf EBSD file format

        Args:
            ctf_file: Path to .ctf file

        Returns:
            Dictionary with EBSD data

        Raises:
            NotImplementedError: CTF parsing is not yet implemented.
        """
        raise NotImplementedError(
            "HKL .ctf EBSD parsing is not yet implemented. "
            "Convert the data to CSV with columns phi1, phi, phi2, x, y "
            "or implement _parse_ctf_file."
        )

    def analyze_microstructure(self, ebsd_data):
        """
        Analyze microstructure from EBSD data

        Args:
            ebsd_data: Dictionary with EBSD data

        Returns:
            Dictionary with microstructure metrics
        """
        # Calculate grain sizes
        grain_sizes, grain_map = self._calculate_grain_sizes(ebsd_data)

        # Calculate texture
        texture_metrics = self._calculate_texture(ebsd_data)

        # Calculate grain orientation spread (GOS)
        gos = self._calculate_grain_orientation_spread(ebsd_data, grain_map)

        # Calculate kernel average misorientation (KAM)
        kam = self._calculate_kam(ebsd_data)

        # Save visualizations
        self._save_ebsd_visualizations(ebsd_data, grain_map, kam)

        # Combine results
        results = {
            'grain_metrics': {
                'mean_grain_size': np.mean(grain_sizes),
                'median_grain_size': np.median(grain_sizes),
                'min_grain_size': np.min(grain_sizes),
                'max_grain_size': np.max(grain_sizes),
                'std_grain_size': np.std(grain_sizes),
                'grain_size_distribution': {
                    'histogram': np.histogram(grain_sizes, bins=20)[0].tolist(),
                    'bin_edges': np.histogram(grain_sizes, bins=20)[1].tolist()
                }
            },
            'texture_metrics': texture_metrics,
            'gos_metrics': {
                'mean_gos': np.mean(gos),
                'median_gos': np.median(gos),
                'max_gos': np.max(gos)
            },
            'kam_metrics': {
                'mean_kam': np.mean(kam),
                'median_kam': np.median(kam),
                'max_kam': np.max(kam)
            }
        }

        # Save results
        with open(self.run_dir / 'microstructure_results.yaml', 'w') as f:
            yaml.dump(results, f)

        logger.info(f"Microstructure analysis completed: mean grain size {results['grain_metrics']['mean_grain_size']:.2f}")
        return results

    def _calculate_grain_sizes(self, ebsd_data):
        """
        Calculate grain sizes from EBSD data

        Args:
            ebsd_data: Dictionary with EBSD data

        Returns:
            Tuple of (grain_sizes, grain_map)
        """
        # This is a simplified implementation
        # In practice, grain segmentation would use misorientation thresholds

        # Check if data is already on a grid
        if ebsd_data['phi1'].ndim == 2:
            # Use simplified approach for demonstration - threshold CI to identify grains
            # In reality, would use misorientation between neighboring pixels

            # Create a synthetic grain map by thresholding confidence index
            if 'ci' in ebsd_data:
                grain_map = measure.label(ebsd_data['ci'] > 0.1)
            else:
                # Random grain map for demonstration
                grain_map = np.random.randint(1, 100, size=ebsd_data['phi1'].shape)

            # Calculate grain properties
            grain_props = measure.regionprops(grain_map)
            grain_sizes = [prop.area for prop in grain_props]

            return grain_sizes, grain_map
        else:
            # For point data, need to grid it first
            logger.warning("Point data needs gridding first. Using dummy grain map.")
            grain_map = np.random.randint(1, 100, size=(100, 100))
            grain_sizes = np.random.lognormal(3, 1, 100)
            return grain_sizes, grain_map

    def _calculate_texture(self, ebsd_data):
        """
        Calculate crystallographic texture from EBSD data

        Args:
            ebsd_data: Dictionary with EBSD data

        Returns:
            Dictionary with texture metrics
        """
        # This is a placeholder - actual texture analysis would use orientation
        # distribution functions, pole figures, etc.

        # Calculate a simple approximation of texture strength
        if ebsd_data['phi1'].ndim == 2:
            # Use standard deviation of Euler angles as a simple texture measure
            texture_strength = (
                np.std(ebsd_data['phi1']) +
                np.std(ebsd_data['phi']) +
                np.std(ebsd_data['phi2'])
            ) / 3
        else:
            texture_strength = (
                np.std(ebsd_data['phi1']) +
                np.std(ebsd_data['phi']) +
                np.std(ebsd_data['phi2'])
            ) / 3

        return {
            'texture_strength': texture_strength,
            # In a real implementation, would include more sophisticated metrics
            'texture_index': np.random.rand() * 10,  # Placeholder
            'j_index': np.random.rand() * 20         # Placeholder
        }

    def _calculate_grain_orientation_spread(self, ebsd_data, grain_map):
        """
        Calculate grain orientation spread (GOS) from EBSD data

        Args:
            ebsd_data: Dictionary with EBSD data
            grain_map: Array with grain labels

        Returns:
            Array with GOS values for each grain
        """
        # This is a placeholder - actual GOS calculation would be more complex
        num_grains = np.max(grain_map)
        return np.random.rand(num_grains) * 5  # Random GOS values in degrees

    def _calculate_kam(self, ebsd_data):
        """
        Calculate kernel average misorientation (KAM) from EBSD data

        Args:
            ebsd_data: Dictionary with EBSD data

        Returns:
            Array with KAM values
        """
        # This is a placeholder - actual KAM calculation would use neighbor misorientations
        if ebsd_data['phi1'].ndim == 2:
            return np.random.rand(*ebsd_data['phi1'].shape) * 2  # Random KAM values in degrees
        else:
            return np.random.rand(len(ebsd_data['phi1'])) * 2

    def _save_ebsd_visualizations(self, ebsd_data, grain_map, kam):
        """
        Save visualizations of EBSD analysis

        Args:
            ebsd_data: Dictionary with EBSD data
            grain_map: Array with grain labels
            kam: Array with KAM values
        """
        # Make sure the data is 2D for visualization
        if ebsd_data['phi1'].ndim != 2:
            logger.warning("Cannot create visualizations for non-gridded data")
            return

        # IPF (Inverse Pole Figure) coloring
        # This is a very simplified IPF coloring scheme for demonstration
        # Real IPF coloring would use quaternion rotations and crystal symmetry

        # Create a simplified IPF color map using Euler angles
        phi1_normalized = ebsd_data['phi1'] / 360
        phi_normalized = ebsd_data['phi'] / 90
        phi2_normalized = ebsd_data['phi2'] / 360

        ipf_colors = np.zeros((*ebsd_data['phi1'].shape, 3))
        ipf_colors[..., 0] = phi1_normalized  # Red channel
        ipf_colors[..., 1] = phi_normalized   # Green channel
        ipf_colors[..., 2] = phi2_normalized  # Blue channel

        # Grain boundary map
        grain_boundaries = np.zeros_like(grain_map, dtype=float)
        for i in range(1, 3):
            for j in range(1, 3):
                shifted = np.roll(grain_map, (i-1, j-1), axis=(0, 1))
                grain_boundaries += (grain_map != shifted)

        grain_boundaries = grain_boundaries > 0

        # Save IPF map
        plt.figure(figsize=(10, 8))
        plt.imshow(ipf_colors)
        plt.title("Simplified IPF Coloring")
        plt.savefig(self.run_dir / 'ipf_map.png')
        plt.close()

        # Save grain map
        plt.figure(figsize=(10, 8))
        plt.imshow(grain_map, cmap='jet')
        plt.title("Grain Map")
        plt.colorbar(label='Grain ID')
        plt.savefig(self.run_dir / 'grain_map.png')
        plt.close()

        # Save KAM map
        plt.figure(figsize=(10, 8))
        plt.imshow(kam, cmap='hot')
        plt.title("Kernel Average Misorientation (KAM)")
        plt.colorbar(label='Misorientation (degrees)')
        plt.savefig(self.run_dir / 'kam_map.png')
        plt.close()

        # Save grain boundaries
        plt.figure(figsize=(10, 8))
        plt.imshow(grain_boundaries, cmap='gray')
        plt.title("Grain Boundaries")
        plt.savefig(self.run_dir / 'grain_boundaries.png')
        plt.close()

    def measure_residual_stress(self, stress_data_file):
        """
        Process residual stress measurements (e.g., from XRD or contour method)

        Args:
            stress_data_file: Path to residual stress data file (CSV)

        Returns:
            Dictionary with residual stress metrics

        Raises:
            ValueError: If the file format is unsupported or required columns
                are missing.
            RuntimeError: If CSV parsing fails for any other reason.
        """
        logger.info(f"Processing residual stress data from {stress_data_file}")

        file_ext = Path(stress_data_file).suffix.lower()

        if file_ext != '.csv':
            raise ValueError(
                f"Unsupported stress data format '{file_ext}' for '{stress_data_file}'. "
                "Only CSV files are supported."
            )

        try:
            stress_df, stress_metrics = parse_stress_csv(stress_data_file)
            von_mises = stress_df['von_mises'].values
        except Exception as e:
            raise RuntimeError(
                f"Failed to process stress CSV '{stress_data_file}': {e}"
            ) from e

        # If spatial coordinates are available, create visualization
        if all(col in stress_df.columns for col in ['x', 'y']):
            x = stress_df['x'].values
            y = stress_df['y'].values

            x_unique = np.unique(x)
            y_unique = np.unique(y)

            if len(x_unique) * len(y_unique) == len(x):
                X, Y = np.meshgrid(x_unique, y_unique)
                VM = von_mises.reshape(len(y_unique), len(x_unique))

                plt.figure(figsize=(10, 8))
                plt.contourf(X, Y, VM, cmap='jet', levels=20)
                plt.colorbar(label='von Mises Stress (MPa)')
                plt.title('Residual Stress Distribution')
                plt.xlabel('X (mm)')
                plt.ylabel('Y (mm)')
                plt.savefig(self.run_dir / 'residual_stress_map.png')
                plt.close()

        # Save results
        with open(self.run_dir / 'residual_stress_results.yaml', 'w') as f:
            yaml.dump(stress_metrics, f)

        logger.info(
            f"Residual stress analysis completed: mean von Mises = "
            f"{stress_metrics['mean_von_mises']:.2f} MPa"
        )
        return stress_metrics

    def compare_with_predictions(self, experimental_data, prediction_file):
        """
        Compare experimental measurements with PINN predictions

        Predictions may be supplied as an HDF5 file with datasets
        ``predictions/stress``, ``predictions/porosity`` and
        ``predictions/geometric_accuracy`` or as a CSV containing the columns
        ``residual_stress``, ``porosity`` and ``geometric_accuracy``. CSV
        prediction files may also contain parameter columns such as ``P`` and
        ``v``.

        Args:
            experimental_data: Dictionary with experimental measurements. The
                method looks for ``stress.mean_von_mises``,
                ``porosity.porosity`` and ``geometric_accuracy`` /
                ``geometric.accuracy``.
            prediction_file: Path to file with PINN predictions (HDF5 or CSV).

        Returns:
            Dictionary with comparison metrics. Each available quantity
            contains ``predicted``, ``experimental``, ``mae``, ``rmse`` and
            ``error_percent`` keys when the experimental value is present.

        Raises:
            ValueError: If the prediction file format is unsupported or the
                required prediction columns/datasets are missing.
            RuntimeError: If the prediction file cannot be loaded.
        """
        # Load predictions
        file_ext = Path(prediction_file).suffix.lower()

        if file_ext in ['.h5', '.hdf5']:
            try:
                with h5py.File(prediction_file, 'r') as f:
                    pred_stress = f['predictions/stress'][:]
                    pred_porosity = f['predictions/porosity'][:]
                    pred_geometric = f['predictions/geometric_accuracy'][:]

                    scan_vectors = f['parameters'][:]
                    param_names = [name.decode('utf-8') for name in f['parameter_names'][:]]
                    scan_params = {name: values for name, values in zip(param_names, scan_vectors.T)}
            except Exception as e:
                raise RuntimeError(f"Failed to load predictions from HDF5 '{prediction_file}': {e}") from e

        elif file_ext == '.csv':
            try:
                pred_df = pd.read_csv(prediction_file)
                required = ['residual_stress', 'porosity', 'geometric_accuracy']
                missing = [col for col in required if col not in pred_df.columns]
                if missing:
                    raise ValueError(
                        f"Prediction CSV '{prediction_file}' is missing required columns: {missing}. "
                        f"Found columns: {list(pred_df.columns)}"
                    )

                pred_stress = pred_df['residual_stress'].values
                pred_porosity = pred_df['porosity'].values
                pred_geometric = pred_df['geometric_accuracy'].values

                scan_params = {
                    col: pred_df[col].values
                    for col in pred_df.columns
                    if col not in required
                }
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise RuntimeError(f"Failed to load predictions from CSV '{prediction_file}': {e}") from e
        else:
            raise ValueError(
                f"Unsupported prediction file format '{file_ext}' for '{prediction_file}'. "
                "Supported formats are .h5, .hdf5 and .csv."
            )

        # Extract experimental values if present
        exp_stress = self._extract_experimental_value(
            experimental_data, 'stress', ['mean_von_mises', 'von_mises', 'mean']
        )
        exp_porosity = self._extract_experimental_value(
            experimental_data, 'porosity', ['porosity', 'mean']
        )
        exp_geometric = self._extract_experimental_value(
            experimental_data,
            'geometric_accuracy',
            ['accuracy', 'geometric_accuracy', 'mean']
        )
        if exp_geometric is None:
            exp_geometric = self._extract_experimental_value(
                experimental_data, 'geometric', ['accuracy', 'geometric_accuracy', 'mean']
            )

        # Build comparison metrics per quantity
        comparison = {}

        if exp_stress is not None:
            comparison['stress'] = self._compute_comparison_metrics(pred_stress, exp_stress)
        if exp_porosity is not None:
            comparison['porosity'] = self._compute_comparison_metrics(pred_porosity, exp_porosity)
        if exp_geometric is not None:
            comparison['geometric_accuracy'] = self._compute_comparison_metrics(
                pred_geometric, exp_geometric
            )

        if not comparison:
            logger.warning(
                "No experimental values found in ``experimental_data`` for comparison."
            )

        # Calculate R² for stress vs. parameters (simplified)
        from sklearn.linear_model import LinearRegression
        if 'P' in scan_params and 'v' in scan_params and len(pred_stress) > 1:
            X = np.column_stack([scan_params['P'], scan_params['v']])
            model = LinearRegression()
            model.fit(X, pred_stress)
            comparison['r2_score'] = float(model.score(X, pred_stress))
        else:
            comparison['r2_score'] = 0.0

        # Save comparison
        with open(self.run_dir / 'prediction_comparison.yaml', 'w') as f:
            yaml.dump(comparison, f)

        # Create visualization
        self._save_prediction_comparison_plot(
            scan_params, pred_stress, pred_porosity,
            exp_stress, exp_porosity
        )

        logger.info("Prediction comparison completed")
        return comparison

    def _extract_experimental_value(self, experimental_data, key, fallback_keys):
        """Extract an experimental scalar or array from ``experimental_data``.

        Args:
            experimental_data: Dictionary of experimental results.
            key: Primary key to look up.
            fallback_keys: List of sub-keys to try inside ``experimental_data[key]``.

        Returns:
            The experimental value (scalar or array) or ``None`` if not found.
        """
        if key not in experimental_data:
            return None

        value = experimental_data[key]
        if isinstance(value, dict):
            for sub_key in fallback_keys:
                if sub_key in value:
                    return value[sub_key]
            return None

        return value

    def _compute_comparison_metrics(self, predicted, experimental):
        """Compute MAE, RMSE and relative error for a single quantity.

        Args:
            predicted: Array of predicted values.
            experimental: Scalar or array of experimental values.

        Returns:
            Dictionary with ``predicted``, ``experimental``, ``mae``, ``rmse``
            and ``error_percent``.
        """
        predicted = np.asarray(predicted)
        experimental = np.asarray(experimental)

        if predicted.ndim == 0:
            predicted = predicted.reshape(1)
        if experimental.ndim == 0:
            experimental = np.full_like(predicted, experimental)

        mae = float(np.mean(np.abs(predicted - experimental)))
        rmse = float(np.sqrt(np.mean((predicted - experimental) ** 2)))
        mean_pred = float(np.mean(predicted))
        mean_exp = float(np.mean(experimental))
        error_percent = 100.0 * abs(mean_pred - mean_exp) / max(abs(mean_exp), 1e-12)

        return {
            'predicted': mean_pred,
            'experimental': mean_exp,
            'mae': mae,
            'rmse': rmse,
            'error_percent': error_percent,
        }

    def _save_prediction_comparison_plot(self, scan_params, pred_stress, pred_porosity,
                                         exp_stress, exp_porosity):
        """Save a four-panel comparison plot of predictions vs. parameters."""
        P = scan_params.get('P', np.array([]))
        v = scan_params.get('v', np.array([]))

        plt.figure(figsize=(12, 8))

        if len(P) > 0:
            plt.subplot(221)
            plt.scatter(P, pred_stress, alpha=0.7)
            if exp_stress is not None:
                plt.axhline(y=np.mean(exp_stress), color='r', linestyle='-', label='Experimental')
            plt.xlabel('Laser Power (W)')
            plt.ylabel('Residual Stress (MPa)')
            plt.title('Stress vs. Power')
            plt.legend()

            plt.subplot(223)
            plt.scatter(P, pred_porosity, alpha=0.7)
            if exp_porosity is not None:
                plt.axhline(y=np.mean(exp_porosity), color='r', linestyle='-', label='Experimental')
            plt.xlabel('Laser Power (W)')
            plt.ylabel('Porosity')
            plt.title('Porosity vs. Power')
            plt.legend()

        if len(v) > 0:
            plt.subplot(222)
            plt.scatter(v, pred_stress, alpha=0.7)
            if exp_stress is not None:
                plt.axhline(y=np.mean(exp_stress), color='r', linestyle='-', label='Experimental')
            plt.xlabel('Scan Speed (mm/s)')
            plt.ylabel('Residual Stress (MPa)')
            plt.title('Stress vs. Speed')
            plt.legend()

            plt.subplot(224)
            plt.scatter(v, pred_porosity, alpha=0.7)
            if exp_porosity is not None:
                plt.axhline(y=np.mean(exp_porosity), color='r', linestyle='-', label='Experimental')
            plt.xlabel('Scan Speed (mm/s)')
            plt.ylabel('Porosity')
            plt.title('Porosity vs. Speed')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.run_dir / 'prediction_comparison.png')
        plt.close()

    def run_full_characterisation(self, xct_file=None, ebsd_file=None, stress_file=None, prediction_file=None):
        """
        Run a complete characterization workflow

        Args:
            xct_file: Path to XCT data file
            ebsd_file: Path to EBSD data file
            stress_file: Path to residual stress data file
            prediction_file: Path to prediction file

        Returns:
            Dictionary with all characterization results
        """
        results = {}

        # Process XCT data if available
        if xct_file:
            logger.info(f"Processing XCT data from {xct_file}")
            file_ext = Path(xct_file).suffix.lower()
            if file_ext == '.csv':
                porosity_df, porosity_summary = parse_xct_porosity_csv(xct_file)
                porosity_results = dict(porosity_summary)
                porosity_results['dataframe'] = porosity_df
                results['porosity'] = porosity_results
            else:
                xct_data = self.load_xct_data(xct_file)
                if xct_data is not None:
                    porosity_results = self.analyze_porosity(xct_data)
                    results['porosity'] = porosity_results

        # Process EBSD data if available
        if ebsd_file:
            logger.info(f"Processing EBSD data from {ebsd_file}")
            ebsd_data = self.load_ebsd_data(ebsd_file)
            if ebsd_data is not None:
                microstructure_results = self.analyze_microstructure(ebsd_data)
                results['microstructure'] = microstructure_results

        # Process residual stress data if available
        if stress_file:
            logger.info(f"Processing residual stress data from {stress_file}")
            stress_results = self.measure_residual_stress(stress_file)
            results['stress'] = stress_results

        # Compare with predictions if available
        if prediction_file and len(results) > 0:
            logger.info(f"Comparing with predictions from {prediction_file}")
            comparison = self.compare_with_predictions(results, prediction_file)
            results['comparison'] = comparison

        # Save all results
        with open(self.run_dir / 'all_results.yaml', 'w') as f:
            yaml.dump(results, f)

        logger.info(f"Full characterization completed, results saved to {self.run_dir}")
        return results


def main():
    """Main function to run characterization"""
    parser = argparse.ArgumentParser(description='Characterize LPBF-built samples')
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--xct', type=str, default=None,
                        help='Path to XCT data file')
    parser.add_argument('--ebsd', type=str, default=None,
                        help='Path to EBSD data file')
    parser.add_argument('--stress', type=str, default=None,
                        help='Path to residual stress data file')
    parser.add_argument('--predictions', type=str, default=None,
                        help='Path to prediction file')

    args = parser.parse_args()

    # Create characterization system
    char = LPBFCharacterisation(args.config)

    # Run characterization
    char.run_full_characterisation(
        xct_file=args.xct,
        ebsd_file=args.ebsd,
        stress_file=args.stress,
        prediction_file=args.predictions
    )

    print(f"Characterization completed. Results saved to {char.run_dir}")


if __name__ == '__main__':
    main()
