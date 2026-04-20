"""
Fast FASTA file parser for sequence length extraction
"""

import os
import logging
from typing import Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FastaLengthParser:
    """
    Fast parser to extract sequence lengths from FASTA files
    """
    
    def __init__(self, fasta_file: str):
        """
        Initialize FASTA parser
        
        Args:
            fasta_file (str): Path to FASTA file
        """
        self.fasta_file = fasta_file
        self.length_cache = {}
        self._parse_file()
    
    def _parse_file(self):
        """Parse FASTA file and extract sequence lengths"""
        if not os.path.exists(self.fasta_file):
            logger.error(f"FASTA file not found: {self.fasta_file}")
            return
        
        logger.info(f"解析FASTA文件: {self.fasta_file}")
        
        # Get file size for progress bar
        file_size = os.path.getsize(self.fasta_file)
        
        current_id = None
        current_length = 0
        processed_proteins = 0
        
        with open(self.fasta_file, 'r') as f:
            # Create progress bar based on file size
            with tqdm(total=file_size, desc="Parsing FASTA", unit='B', unit_scale=True) as pbar:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        # Save previous protein if exists
                        if current_id is not None and current_length > 0:
                            self.length_cache[current_id] = current_length
                            processed_proteins += 1
                        
                        # Extract protein ID (everything after '>')
                        current_id = line[1:].split()[0]  # Take first part before space
                        current_length = 0
                    else:
                        # Add sequence length
                        current_length += len(line)
                
                # Save last protein
                if current_id is not None and current_length > 0:
                    self.length_cache[current_id] = current_length
                    processed_proteins += 1
        
        logger.info(f"已解析 {processed_proteins} 个蛋白质序列")
        logger.info(f"长度范围: {min(self.length_cache.values())}-{max(self.length_cache.values())} 个氨基酸")
    
    def get_length(self, protein_id: str) -> Optional[int]:
        """
        Get sequence length for a protein
        
        Args:
            protein_id (str): Protein identifier
            
        Returns:
            int: Sequence length or None if not found
        """
        return self.length_cache.get(protein_id)
    
    def get_lengths(self, protein_ids: list) -> Dict[str, int]:
        """
        Get sequence lengths for multiple proteins
        
        Args:
            protein_ids (list): List of protein identifiers
            
        Returns:
            Dict[str, int]: Dictionary mapping protein IDs to lengths
        """
        lengths = {}
        missing = []
        
        for protein_id in protein_ids:
            length = self.get_length(protein_id)
            if length is not None:
                lengths[protein_id] = length
            else:
                missing.append(protein_id)
        
        if missing:
            logger.warning(f"{len(missing)} proteins not found in FASTA: {missing[:5]}...")
        
        return lengths
    
    def get_stats(self) -> Dict:
        """Get parser statistics"""
        if not self.length_cache:
            return {'total_proteins': 0}
        
        lengths = list(self.length_cache.values())
        return {
            'total_proteins': len(self.length_cache),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'median_length': sorted(lengths)[len(lengths)//2]
        }


def parse_fasta_lengths(fasta_file: str) -> Dict[str, int]:
    """
    Parse FASTA file and return protein lengths
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        Dict[str, int]: Dictionary mapping protein IDs to sequence lengths
    """
    parser = FastaLengthParser(fasta_file)
    return parser.length_cache


def parse_fasta_file(fasta_file: str) -> Dict[str, str]:
    """
    Parse FASTA file and return protein sequences
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        Dict[str, str]: Dictionary mapping protein IDs to sequences
    """
    sequences = {}
    
    if not os.path.exists(fasta_file):
        logger.error(f"FASTA file not found: {fasta_file}")
        return sequences
    
    logger.info(f"解析FASTA文件获取序列: {fasta_file}")
    
    # Get file size for progress bar
    file_size = os.path.getsize(fasta_file)
    
    current_id = None
    current_seq = []
    processed_proteins = 0
    
    with open(fasta_file, 'r') as f:
        # Create progress bar based on file size
        with tqdm(total=file_size, desc="Parsing sequences", unit='B', unit_scale=True) as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                        processed_proteins += 1
                    
                    # Start new sequence
                    header = line[1:]
                    # Extract protein ID (first part before space)
                    current_id = header.split()[0] if header else None
                    current_seq = []
                else:
                    # Accumulate sequence
                    current_seq.append(line)
            
            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
                processed_proteins += 1
    
    logger.info(f"解析完成: {processed_proteins} 个蛋白质序列")
    return sequences


def create_length_cache(fasta_file: str, protein_ids: list) -> Dict[str, int]:
    """
    Create length cache for specific protein IDs
    
    Args:
        fasta_file (str): Path to FASTA file
        protein_ids (list): List of protein IDs to cache
        
    Returns:
        Dict[str, int]: Length cache for specified proteins
    """
    parser = FastaLengthParser(fasta_file)
    return parser.get_lengths(protein_ids)