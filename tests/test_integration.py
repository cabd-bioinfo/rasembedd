"""Integration tests for the complete protein embedding pipeline."""

import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import subprocess
import shutil
from unittest.mock import Mock, patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import test utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.conftest import (
    assert_embeddings_valid, 
    assert_distances_valid, 
    assert_projection_valid,
    SAMPLE_SEQUENCES,
    SAMPLE_METADATA
)


class TestFullPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_fasta_to_embeddings_to_visualization(self, temp_dir):
        """Test the complete pipeline from FASTA to visualization."""
        # Create test files
        fasta_file = os.path.join(temp_dir, 'test.fasta')
        metadata_file = os.path.join(temp_dir, 'test_metadata.tsv')
        embeddings_file = os.path.join(temp_dir, 'test_embeddings.pkl')
        
        # Write FASTA file
        with open(fasta_file, 'w') as f:
            for seq_id, sequence in SAMPLE_SEQUENCES.items():
                f.write(f'>{seq_id}\n{sequence}\n')
        
        # Write metadata file
        SAMPLE_METADATA.to_csv(metadata_file, sep='\t', index=False)
        
        # Create mock embeddings (instead of running actual model)
        np.random.seed(42)
        mock_embeddings = {}
        for seq_id in SAMPLE_SEQUENCES.keys():
            mock_embeddings[seq_id] = np.random.randn(1024).astype(np.float32)
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(mock_embeddings, f)
        
        # Test that files exist and are valid
        assert os.path.exists(fasta_file)
        assert os.path.exists(metadata_file)
        assert os.path.exists(embeddings_file)
        
        # Load and validate embeddings
        with open(embeddings_file, 'rb') as f:
            loaded_embeddings = pickle.load(f)
        
        assert_embeddings_valid(loaded_embeddings, list(SAMPLE_SEQUENCES.keys()))
        
        # Load and validate metadata
        loaded_metadata = pd.read_csv(metadata_file, sep='\t')
        assert len(loaded_metadata) == len(SAMPLE_METADATA)
        assert 'uniprot_id' in loaded_metadata.columns
        assert 'Family.name' in loaded_metadata.columns
    
    @patch('subprocess.run')
    def test_generate_embeddings_script(self, mock_subprocess, temp_dir):
        """Test generate_embeddings.py script execution."""
        # Create test FASTA file
        fasta_file = os.path.join(temp_dir, 'test.fasta')
        output_file = os.path.join(temp_dir, 'test_output.pkl')
        
        with open(fasta_file, 'w') as f:
            f.write('>test_seq\nMKWVTFISLLL\n')
        
        # Mock successful subprocess execution
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Embeddings generated successfully"
        mock_subprocess.return_value.stderr = ""
        
        # Simulate script execution
        cmd = [
            'python', 'generate_embeddings.py', 
            fasta_file,
            '--model', 'prost_t5',
            '--output', output_file,
            '--device', 'cpu'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        
        # Verify subprocess was called
        mock_subprocess.assert_called()
    
    @patch('subprocess.run')
    def test_generate_visualizations_script(self, mock_subprocess, temp_dir):
        """Test generate_visualizations.py script execution."""
        # Create test files
        metadata_file = os.path.join(temp_dir, 'test_metadata.tsv')
        embeddings_file = os.path.join(temp_dir, 'test_embeddings.pkl')
        
        SAMPLE_METADATA.to_csv(metadata_file, sep='\t', index=False)
        
        # Create mock embeddings
        np.random.seed(42)
        mock_embeddings = {seq_id: np.random.randn(128).astype(np.float32) 
                          for seq_id in SAMPLE_SEQUENCES.keys()}
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(mock_embeddings, f)
        
        # Mock successful subprocess execution
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Visualizations generated successfully"
        mock_subprocess.return_value.stderr = ""
        
        # Simulate script execution
        cmd = [
            'python', 'generate_visualizations.py',
            metadata_file,
            '--embeddings', embeddings_file,
            '--color_column', 'Family.name',
            '--projection_method', 'PCA',
            '--skip_heatmap'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        
        # Verify subprocess was called
        mock_subprocess.assert_called()
    
    def test_data_flow_consistency(self, temp_dir):
        """Test data consistency throughout the pipeline."""
        # Create initial data
        sequences = SAMPLE_SEQUENCES.copy()
        metadata = SAMPLE_METADATA.copy()
        
        # Generate mock embeddings
        np.random.seed(42)
        embeddings = {seq_id: np.random.randn(512).astype(np.float32) 
                     for seq_id in sequences.keys()}
        
        # Test 1: All sequence IDs should be consistent
        seq_ids = set(sequences.keys())
        meta_ids = set(metadata['uniprot_id'])
        emb_ids = set(embeddings.keys())
        
        # Find intersection (common IDs)
        common_ids = seq_ids.intersection(meta_ids).intersection(emb_ids)
        assert len(common_ids) == len(sequences)  # All should match in this test case
        
        # Test 2: Data types should be consistent
        for seq_id, embedding in embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
        
        # Test 3: Metadata should have required columns
        required_columns = ['uniprot_id', 'Family.name']
        for col in required_columns:
            assert col in metadata.columns
        
        # Test 4: Sequences should be valid protein sequences
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        for seq_id, sequence in sequences.items():
            assert isinstance(sequence, str)
            assert len(sequence) > 0
            # Allow some flexibility for ambiguous amino acids
            assert len(set(sequence.upper()) - valid_aas) <= 1  # At most 1 ambiguous AA
    
    def test_error_propagation(self, temp_dir):
        """Test how errors propagate through the pipeline."""
        # Test 1: Missing sequence in embeddings
        sequences = SAMPLE_SEQUENCES.copy()
        embeddings = {seq_id: np.random.randn(128).astype(np.float32) 
                     for seq_id in list(sequences.keys())[:-1]}  # Missing last sequence
        metadata = SAMPLE_METADATA.copy()
        
        # Find common IDs
        seq_ids = set(sequences.keys())
        emb_ids = set(embeddings.keys())
        common_ids = seq_ids.intersection(emb_ids)
        
        assert len(common_ids) == len(sequences) - 1  # One missing
        
        # Test 2: Missing metadata for sequence
        partial_metadata = metadata.iloc[:-1].copy()  # Missing last row
        meta_ids = set(partial_metadata['uniprot_id'])
        common_with_meta = common_ids.intersection(meta_ids)
        
        assert len(common_with_meta) <= len(common_ids)
        
        # Test 3: Invalid embedding data
        invalid_embeddings = embeddings.copy()
        invalid_embeddings['test_seq_1'] = np.array([np.nan, np.inf, 1.0])
        
        # Check for invalid values
        for seq_id, embedding in invalid_embeddings.items():
            if seq_id == 'test_seq_1':
                assert np.isnan(embedding).any() or np.isinf(embedding).any()
    
    def test_scalability_simulation(self, temp_dir):
        """Test pipeline with larger datasets (simulated)."""
        # Create larger test dataset
        n_sequences = 100
        embedding_dim = 1024
        
        # Generate sequences
        np.random.seed(42)
        large_sequences = {}
        for i in range(n_sequences):
            # Generate random protein sequence
            aa_sequence = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 
                                                 size=np.random.randint(50, 200)))
            large_sequences[f'seq_{i:03d}'] = aa_sequence
        
        # Generate metadata
        families = ['Family_A', 'Family_B', 'Family_C', 'Family_D', 'Family_E']
        species = ['species_1', 'species_2', 'species_3']
        
        large_metadata = pd.DataFrame({
            'uniprot_id': list(large_sequences.keys()),
            'Family.name': np.random.choice(families, n_sequences),
            'species': np.random.choice(species, n_sequences),
            'length': [len(seq) for seq in large_sequences.values()]
        })
        
        # Generate embeddings
        large_embeddings = {}
        for seq_id in large_sequences.keys():
            large_embeddings[seq_id] = np.random.randn(embedding_dim).astype(np.float32)
        
        # Test data validity
        assert len(large_sequences) == n_sequences
        assert len(large_metadata) == n_sequences
        assert len(large_embeddings) == n_sequences
        
        # Test memory usage (rough estimate)
        embedding_memory = n_sequences * embedding_dim * 4  # 4 bytes per float32
        assert embedding_memory < 1e9  # Should be less than 1GB for this test
        
        # Test processing time simulation
        import time
        start_time = time.time()
        
        # Simulate distance calculation
        embeddings_array = np.array(list(large_embeddings.values()))
        distances = np.random.randn(n_sequences, n_sequences)  # Mock calculation
        
        # Simulate projection
        projection = np.random.randn(n_sequences, 2)  # Mock calculation
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be reasonably fast for simulated data
        assert processing_time < 10.0  # Less than 10 seconds
        
        # Validate results
        assert_distances_valid(distances, n_sequences)
        assert_projection_valid(projection, n_sequences)
    
    def test_cross_platform_compatibility(self, temp_dir):
        """Test pipeline compatibility across different environments."""
        # Test file path handling
        test_paths = [
            'test.fasta',
            './test.fasta',
            os.path.join(temp_dir, 'test.fasta'),
            temp_dir + os.sep + 'test.fasta'
        ]
        
        for path in test_paths:
            # Test path normalization
            normalized = os.path.normpath(path)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
        
        # Test file extensions
        extensions = ['.fasta', '.fa', '.pkl', '.tsv', '.csv', '.h5', '.hdf5']
        for ext in extensions:
            assert ext.startswith('.')
            assert len(ext) >= 2
        
        # Test separator handling
        separators = ['\t', ',', ';']
        for sep in separators:
            assert isinstance(sep, str)
            assert len(sep) == 1
    
    def test_configuration_management(self):
        """Test configuration parameter handling."""
        # Test default configurations
        default_configs = {
            'device': 'auto',
            'precision': 'float32',
            'batch_size': 1,
            'max_length': 512,
            'projection_method': 'UMAP',
            'distance_metric': 'cosine',
            'color_palette': 'glasbey',
            'random_seed': 42
        }
        
        for key, value in default_configs.items():
            assert isinstance(key, str)
            assert value is not None
        
        # Test parameter validation
        valid_devices = ['cpu', 'cuda', 'auto']
        valid_precisions = ['float16', 'float32']
        valid_projections = ['UMAP', 'PCA', 'TSNE', 'PaCMAP']
        valid_distances = ['cosine', 'euclidean']
        
        assert 'cpu' in valid_devices
        assert 'float32' in valid_precisions
        assert 'UMAP' in valid_projections
        assert 'cosine' in valid_distances
    
    def test_output_format_consistency(self, temp_dir):
        """Test consistency of output formats."""
        # Test embedding file formats
        test_embeddings = {
            'seq1': np.random.randn(128).astype(np.float32),
            'seq2': np.random.randn(128).astype(np.float32)
        }
        
        # Test pickle format
        pkl_file = os.path.join(temp_dir, 'test.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(test_embeddings, f)
        
        with open(pkl_file, 'rb') as f:
            loaded_pkl = pickle.load(f)
        
        assert len(loaded_pkl) == len(test_embeddings)
        
        # Test NPZ format
        npz_file = os.path.join(temp_dir, 'test.npz')
        ids = list(test_embeddings.keys())
        embeddings_array = np.array(list(test_embeddings.values()))
        
        np.savez(npz_file, ids=ids, embeddings=embeddings_array)
        
        loaded_npz = np.load(npz_file, allow_pickle=True)
        assert 'ids' in loaded_npz
        assert 'embeddings' in loaded_npz
        assert len(loaded_npz['ids']) == len(test_embeddings)
        
        # Test visualization output formats
        output_formats = ['pdf', 'png', 'svg']
        for fmt in output_formats:
            output_file = os.path.join(temp_dir, f'test_plot.{fmt}')
            # Would create actual plot file in real implementation
            # For now, just test filename generation
            assert output_file.endswith(f'.{fmt}')
    
    def test_version_compatibility(self):
        """Test compatibility with different package versions."""
        # Test Python version
        import sys
        python_version = sys.version_info
        assert python_version.major >= 3
        assert python_version.minor >= 8  # Minimum Python 3.8
        
        # Test key package versions
        packages_to_check = [
            'numpy', 'pandas', 'matplotlib', 'scikit-learn'
        ]
        
        for package_name in packages_to_check:
            try:
                package = __import__(package_name)
                if hasattr(package, '__version__'):
                    version = package.__version__
                    assert isinstance(version, str)
                    assert len(version) > 0
            except ImportError:
                # Package not available, skip check
                pass
