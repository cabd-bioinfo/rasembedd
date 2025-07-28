#!/usr/bin/env python3
"""
Unified script to generate protein embeddings using various protein language models.
This script supports multiple models with different preprocessing requirements.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import pickle
import os
import sys
from typing import Dict, Any, Optional, Tuple
import logging
import importlib

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

# Add models directory to Python path
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if models_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# Import base model class
from models.base_model import BaseEmbeddingModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sequences_from_fasta(fasta_file: str) -> Dict[str, str]:
    """Load sequences from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            elif current_id:
                current_seq.append(line)
        
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

def load_sequences_from_tsv(tsv_file: str, id_column: str = 'uniprot_id', 
                           seq_column: str = 'sequence') -> Dict[str, str]:
    """Load sequences from a TSV file."""
    df = pd.read_csv(tsv_file, sep='\t')
    sequences = pd.Series(df[seq_column].values, index=df[id_column]).to_dict()
    return sequences

def save_embeddings(embeddings: Dict[str, np.ndarray], output_file: str, 
                   format: str = 'pickle') -> None:
    """Save embeddings to file in various formats."""
    if format == 'pickle':
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
    elif format == 'npz':
        ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id] for id in ids])
        np.savez(output_file, embeddings=embedding_matrix, ids=ids)
    elif format == 'hdf5' or format == 'h5':
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        
        with h5py.File(output_file, 'w') as f:
            embedding_group = f.create_group('embeddings')
            metadata_group = f.create_group('metadata')
            
            for seq_id, embedding in embeddings.items():
                safe_id = seq_id.replace('/', '_').replace('\\', '_').replace('|', '_')
                embedding_group.create_dataset(safe_id, data=embedding)
            
            ids = list(embeddings.keys())
            embedding_dim = next(iter(embeddings.values())).shape[0]
            
            ids_bytes = [id.encode('utf-8') for id in ids]
            metadata_group.create_dataset('sequence_ids', data=ids_bytes)
            metadata_group.create_dataset('embedding_dimension', data=embedding_dim)
            metadata_group.create_dataset('num_sequences', data=len(embeddings))
            
            f.attrs['format_version'] = '1.0'
            f.attrs['description'] = 'Protein sequence embeddings'
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Embeddings saved to {output_file}")

def get_available_models() -> Dict[str, str]:
    """Get list of available model implementations."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    available_models = {}
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.py') and not file.startswith('__') and file != 'base_model.py':
                model_name = file[:-3]  # Remove .py extension
                available_models[model_name] = os.path.join(models_dir, file)
    
    return available_models

def load_model_class(model_type: str) -> BaseEmbeddingModel:
    """Dynamically load a model class."""
    try:
        module = importlib.import_module(f'models.{model_type}')
        # Look for a class that ends with 'Model'
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseEmbeddingModel) and 
                attr != BaseEmbeddingModel):
                return attr
        raise ValueError(f"No valid model class found in models.{model_type}")
    except ImportError as e:
        available_models = list(get_available_models().keys())
        raise ValueError(f"Model type '{model_type}' not found. Available models: {available_models}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading model class for '{model_type}': {str(e)}")

def main():
    """Main function to generate protein embeddings."""
    parser = argparse.ArgumentParser(
        description='Generate protein embeddings using various protein language models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument('--input', required=True, 
                       help='Path to input file (FASTA or TSV)')
    parser.add_argument('--input_type', choices=['fasta', 'tsv'], default='tsv',
                       help='Type of input file')
    parser.add_argument('--id_column', default='uniprot_id',
                       help='Column name for sequence IDs (TSV only)')
    parser.add_argument('--seq_column', default='sequence',
                       help='Column name for sequences (TSV only)')
    
    # Model arguments
    parser.add_argument('--model_type', required=True,
                       help='Type of model to use (e.g., prot_t5, esm, custom)')
    parser.add_argument('--model_name', 
                       help='Specific model name/path (model-specific)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cuda, or cpu)')
    
    # Output arguments
    parser.add_argument('--output', required=True,
                       help='Path to output file for embeddings')
    parser.add_argument('--format', choices=['pickle', 'npz', 'hdf5', 'h5'], default='pickle',
                       help='Output format for embeddings')
    
    # Additional model-specific arguments can be passed as --model_args
    parser.add_argument('--model_args', nargs='*', default=[],
                       help='Additional model-specific arguments (key=value pairs)')
    
    args = parser.parse_args()
    
    # Parse model-specific arguments
    model_kwargs = {}
    for arg in args.model_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
            model_kwargs[key] = value
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load sequences
    logger.info(f"Loading sequences from {args.input}")
    if args.input_type == 'fasta':
        sequences = load_sequences_from_fasta(args.input)
    else:
        sequences = load_sequences_from_tsv(args.input, args.id_column, args.seq_column)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Load model
    logger.info(f"Loading model: {args.model_type}")
    try:
        model_class = load_model_class(args.model_type)
        model = model_class(
            model_name=args.model_name,
            device=device,
            **model_kwargs
        )
        model.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.generate_embeddings(sequences)
    
    if not embeddings:
        logger.error("No embeddings were generated")
        return
    
    # Save embeddings
    save_embeddings(embeddings, args.output, args.format)
    
    # Print summary
    embedding_dim = next(iter(embeddings.values())).shape[0]
    logger.info(f"Generated {len(embeddings)} embeddings with dimension {embedding_dim}")
    logger.info("Done!")

if __name__ == "__main__":
    main()
