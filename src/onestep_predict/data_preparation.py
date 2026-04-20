"""
Data Preparation Module for One-Step Prediction

Handles PDB file parsing, sequence extraction, validation, and FASTA generation.
Provides prompts for multifeature generation when needed.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging

# Try importing BioPython for PDB parsing
try:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    PDBParser = None
    seq1 = None
    SeqIO = None


# Standard amino acids (3-letter codes)
STANDARD_AA_3LETTER = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# 3-letter to 1-letter mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


class DataPreparator:
    """
    Prepare prediction data from PDB files and interaction lists.
    
    Responsibilities:
    - Extract protein names and sequences from PDB files
    - Validate sequences for non-standard amino acids
    - Check interaction list against available PDB files
    - Generate FASTA file for downstream processing
    - Prompt user for multifeature generation if needed
    """
    
    def __init__(self, pdb_dir: Optional[str], interaction_list: Optional[str], output_dir: str,
                 genome_mode: bool = False, logger: Optional[logging.Logger] = None,
                 fasta_path: Optional[str] = None,
                 max_sequence_length: int = 1022):
        """
        Initialize the data preparator.
        
        Args:
            pdb_dir: Directory containing PDB files (optional in fasta mode)
            interaction_list: Path to CSV file with protein pairs (and optional labels); optional in genome_mode
            output_dir: Output directory for generated files
            genome_mode: When True, generate upper-triangular pairs on the fly (no pairs.csv)
            logger: Optional logger instance
            fasta_path: Optional input FASTA file for sequence-only mode
            max_sequence_length: Max allowed sequence length in genome mode. Longer proteins are excluded.
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython is required for PDB parsing. "
                "Install it with: pip install biopython"
            )
        
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.interaction_list = Path(interaction_list) if interaction_list else None
        self.fasta_path = Path(fasta_path) if fasta_path else None
        self.output_dir = Path(output_dir)
        self.genome_mode = genome_mode
        self.max_sequence_length = max_sequence_length
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directories
        self.err_dir = self.output_dir / "err"
        self.err_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.protein_sequences: Dict[str, str] = {}
        self.protein_pairs: List[Tuple[str, str]] = []
        self.true_labels: Optional[List[int]] = None
        self.has_labels = False
        self.protein_ids: List[str] = []
        
        # Error tracking
        self.invalid_proteins: Dict[str, str] = {}  # protein_id -> error reason
        self.invalid_pairs: List[Tuple[str, str, str]] = []  # (prot1, prot2, reason)
        self.nonstandard_aa_proteins: Dict[str, List[str]] = {}  # protein_id -> list of nonstandard AAs
        self.excluded_by_length: Dict[str, int] = {}  # protein_id -> sequence length
        self._reused_fasta_file: Optional[Path] = None
        
    def prepare(self) -> Dict[str, any]:
        """
        Run the complete data preparation pipeline.
        
        Returns:
            Dict containing:
                - protein_pairs: List of valid protein pairs
                - protein_sequences: Dict of protein_id -> sequence
                - fasta_file: Path to generated FASTA file
                - has_labels: Whether true labels are available
                - true_labels: List of labels if available
                - requires_multifeature: Whether multifeature generation is needed
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting data preparation...")
        self.logger.info("=" * 60)
        
        # Step 1: Load sequences from FASTA or PDB.
        if self.fasta_path is not None:
            self._load_sequences_from_input_fasta()
        else:
            # Fast path: if proteins.fasta already exists and count matches PDB count,
            # reuse it directly and skip expensive PDB parsing.
            if not self._try_reuse_existing_fasta():
                self._scan_pdb_directory()
        
        # Step 2: Parse interaction list or record genome-mode proteins
        if self.genome_mode:
            self._prepare_genome_mode()
        else:
            self._parse_interaction_list()
            # Step 3: Validate protein pairs
            self._validate_protein_pairs()

        if not self.protein_ids:
            self.protein_ids = sorted(self.protein_sequences.keys())
        
        # Step 4: Save error reports
        self._save_error_reports()
        
        # Step 5: Generate FASTA file
        if self._reused_fasta_file is not None:
            fasta_file = self._reused_fasta_file
            self.logger.info(f"Reusing existing FASTA file: {fasta_file}")
        else:
            fasta_file = self._generate_fasta()
        
        # Step 6: Check for multifeature requirement
        requires_multifeature = self._check_multifeature_requirement()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Data preparation complete!")
        self.logger.info(f"  Valid protein pairs: {len(self.protein_pairs)}")
        self.logger.info(f"  Unique proteins: {len(self.protein_sequences)}")
        self.logger.info(f"  Has labels: {self.has_labels}")
        self.logger.info("=" * 60)
        
        return {
            'protein_pairs': self.protein_pairs,
            'protein_sequences': self.protein_sequences,
            'fasta_file': str(fasta_file),
            'has_labels': self.has_labels,
            'true_labels': self.true_labels,
            'requires_multifeature': requires_multifeature,
            'protein_ids': self.protein_ids,
        }

    def _try_reuse_existing_fasta(self) -> bool:
        """
        Try to reuse existing output FASTA file for faster startup.

        If `output_dir/proteins.fasta` already exists and can be parsed, reuse it
        directly and skip expensive PDB scanning. This is intentionally permissive
        for repeated prediction runs on the same output folder.

        Returns:
            True if existing FASTA was reused successfully, otherwise False.
        """
        fasta_file = self.output_dir / "proteins.fasta"
        if not fasta_file.exists():
            return False

        sequences = self._load_fasta_sequences(fasta_file)
        if not sequences:
            self.logger.info(
                "Existing FASTA is empty or invalid, "
                f"rebuilding from PDB"
            )
            return False

        # In pair-list mode, ensure reused FASTA covers all proteins referenced
        # by the current interaction list. This prevents stale FASTA reuse from
        # silently dropping valid pairs in subsequent runs.
        if (not self.genome_mode) and self.interaction_list is not None and self.interaction_list.exists():
            required_ids = self._collect_required_protein_ids_from_interaction_list(self.interaction_list)
            missing_ids = [pid for pid in required_ids if pid not in sequences]
            if missing_ids:
                self.logger.info(
                    "Existing FASTA does not cover current interaction list "
                    f"(missing {len(missing_ids)} proteins), rebuilding from PDB"
                )
                return False

        self.protein_sequences = sequences
        if self.genome_mode:
            before_count = len(self.protein_sequences)
            self._apply_genome_length_filter()
            if len(self.protein_sequences) != before_count:
                self.logger.info(
                    "Existing FASTA contains over-length proteins for genome mode, "
                    "rebuilding from PDB"
                )
                return False
        self._reused_fasta_file = fasta_file
        self.logger.info(
            f"Detected existing FASTA ({len(sequences)} proteins); "
            f"skipping PDB sequence extraction for faster startup"
        )
        return True

    def _load_sequences_from_input_fasta(self) -> None:
        """Load protein sequences from user-provided FASTA."""
        if self.fasta_path is None or (not self.fasta_path.exists()):
            raise FileNotFoundError(f"Input FASTA not found: {self.fasta_path}")

        sequences = self._load_fasta_sequences(self.fasta_path)
        if not sequences:
            raise ValueError(f"No valid sequences found in FASTA: {self.fasta_path}")

        self.protein_sequences = sequences
        if self.genome_mode:
            self._apply_genome_length_filter()
        self.logger.info(
            f"Loaded {len(self.protein_sequences)} protein sequences from FASTA: {self.fasta_path}"
        )

    def _apply_genome_length_filter(self) -> None:
        """Apply maximum sequence length filter in genome mode."""
        if (not self.genome_mode) or (self.max_sequence_length is None):
            return

        kept_sequences: Dict[str, str] = {}
        removed_count = 0
        for protein_id, sequence in self.protein_sequences.items():
            seq_len = len(sequence)
            if seq_len > self.max_sequence_length:
                self.excluded_by_length[protein_id] = seq_len
                self.invalid_proteins[protein_id] = (
                    f"Sequence length {seq_len} exceeds --lenth/--length limit {self.max_sequence_length}"
                )
                removed_count += 1
                continue
            kept_sequences[protein_id] = sequence

        self.protein_sequences = kept_sequences
        if removed_count > 0:
            self.logger.warning(
                "Genome mode length filter removed "
                f"{removed_count} proteins longer than {self.max_sequence_length}"
            )

    def _collect_required_protein_ids_from_interaction_list(self, interaction_file: Path) -> Set[str]:
        """Collect unique protein IDs from the first two columns of interaction CSV."""
        required: Set[str] = set()
        with open(interaction_file, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                prot1 = parts[0].strip()
                prot2 = parts[1].strip()
                if prot1:
                    required.add(prot1)
                if prot2:
                    required.add(prot2)
        return required

    def _count_fasta_entries(self, fasta_file: Path) -> int:
        """Count number of FASTA records quickly by counting header lines."""
        count = 0
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count

    def _load_fasta_sequences(self, fasta_file: Path) -> Dict[str, str]:
        """Load sequences from FASTA file into protein_id -> sequence mapping."""
        sequences: Dict[str, str] = {}
        current_id: Optional[str] = None
        current_seq_parts: List[str] = []

        with open(fasta_file, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq_parts)
                    current_id = line[1:].split()[0]
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)

        if current_id is not None:
            sequences[current_id] = ''.join(current_seq_parts)

        return sequences
    
    def _scan_pdb_directory(self) -> None:
        """Scan PDB directory and extract protein sequences."""
        if self.pdb_dir is None:
            raise ValueError("pdb_dir is required for PDB scanning mode")

        self.logger.info(f"Scanning PDB directory: {self.pdb_dir}")
        
        if not self.pdb_dir.exists():
            raise FileNotFoundError(f"PDB directory not found: {self.pdb_dir}")
        
        pdb_files = list(self.pdb_dir.glob("*.pdb")) + list(self.pdb_dir.glob("*.PDB"))
        self.logger.info(f"Found {len(pdb_files)} PDB files")
        
        parser = PDBParser(QUIET=True)
        
        for pdb_file in pdb_files:
            protein_id = pdb_file.stem
            
            try:
                sequence, nonstandard_aas = self._extract_sequence_from_pdb(parser, pdb_file)
                
                if sequence:
                    if self.genome_mode and len(sequence) > self.max_sequence_length:
                        self.excluded_by_length[protein_id] = len(sequence)
                        self.invalid_proteins[protein_id] = (
                            f"Sequence length {len(sequence)} exceeds --lenth/--length limit {self.max_sequence_length}"
                        )
                        continue
                    self.protein_sequences[protein_id] = sequence
                    
                    if nonstandard_aas:
                        self.nonstandard_aa_proteins[protein_id] = nonstandard_aas
                        self.logger.warning(
                            f"Protein {protein_id} contains non-standard amino acids: {nonstandard_aas}"
                        )
                else:
                    self.invalid_proteins[protein_id] = "No valid amino acid sequence extracted"
                    
            except Exception as e:
                self.invalid_proteins[protein_id] = f"PDB parsing error: {str(e)}"
                self.logger.error(f"Failed to parse PDB file {pdb_file}: {e}")
        
        self.logger.info(f"Successfully extracted {len(self.protein_sequences)} protein sequences")
        
        if self.invalid_proteins:
            self.logger.warning(f"Failed to extract {len(self.invalid_proteins)} proteins")
    
    def _extract_sequence_from_pdb(self, parser: 'PDBParser', pdb_file: Path) -> Tuple[str, List[str]]:
        """
        Extract amino acid sequence from PDB file.
        
        Args:
            parser: BioPython PDB parser
            pdb_file: Path to PDB file
            
        Returns:
            Tuple of (sequence, list of non-standard amino acids found)
        """
        structure = parser.get_structure(pdb_file.stem, str(pdb_file))
        
        sequence_parts = []
        nonstandard_aas = set()
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname().upper()
                    
                    # Skip water and other non-amino acid residues
                    if residue.id[0] != ' ':  # Hetero atoms
                        continue
                    
                    if res_name in STANDARD_AA_3LETTER:
                        sequence_parts.append(AA_3TO1[res_name])
                    elif res_name in ['MSE', 'SEP', 'TPO', 'PTR']:
                        # Common modified amino acids - map to standard
                        mapping = {'MSE': 'M', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y'}
                        sequence_parts.append(mapping.get(res_name, 'X'))
                        nonstandard_aas.add(res_name)
                    else:
                        # Unknown residue
                        nonstandard_aas.add(res_name)
            
            # Only use first model
            break
        
        sequence = ''.join(sequence_parts)
        return sequence, list(nonstandard_aas)
    
    def _parse_interaction_list(self) -> None:
        """Parse the interaction list CSV file."""
        if self.interaction_list is None:
            raise ValueError("interaction_list is required when genome_mode is False")

        self.logger.info(f"Parsing interaction list: {self.interaction_list}")

        if not self.interaction_list.exists():
            raise FileNotFoundError(f"Interaction list not found: {self.interaction_list}")
        
        pairs = []
        labels = []
        
        with open(self.interaction_list, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                
                if len(parts) < 2:
                    self.logger.warning(f"Line {line_num}: Invalid format, skipping: {line}")
                    continue
                
                prot1 = parts[0].strip()
                prot2 = parts[1].strip()
                
                pairs.append((prot1, prot2))
                
                # Check for labels (third column)
                if len(parts) >= 3:
                    try:
                        label = int(parts[2].strip())
                        labels.append(label)
                    except ValueError:
                        labels.append(None)
        
        self.protein_pairs = pairs
        
        # Check if labels are consistently available
        if labels and all(l is not None for l in labels):
            self.true_labels = labels
            self.has_labels = True
            self.logger.info(f"Found {len(labels)} labels in interaction list")
        else:
            self.true_labels = None
            self.has_labels = False
        
        self.logger.info(f"Parsed {len(pairs)} protein pairs")

    def _prepare_genome_mode(self) -> None:
        """Prepare genome mode metadata without materializing pair list."""
        self.protein_ids = sorted(self.protein_sequences.keys())

        if not self.protein_ids:
            raise ValueError("No valid protein sequences found to generate pairs")

        self.protein_pairs = []  # Not used in genome mode
        self.true_labels = None
        self.has_labels = False
        total_pairs = len(self.protein_ids) * (len(self.protein_ids) + 1) // 2
        self.logger.info(
            f"Genome mode: will scan upper-triangular pairs on the fly ({total_pairs} combinations)"
        )
        self.logger.info(
            f"Genome mode max length: {self.max_sequence_length}"
        )
        if self.excluded_by_length:
            self.logger.info(
                f"Excluded over-length proteins: {len(self.excluded_by_length)}"
            )
    
    def _validate_protein_pairs(self) -> None:
        """Validate protein pairs against available PDB files."""
        self.logger.info("Validating protein pairs...")
        
        valid_pairs = []
        valid_labels = [] if self.has_labels else None
        
        available_proteins = set(self.protein_sequences.keys())
        
        for i, (prot1, prot2) in enumerate(self.protein_pairs):
            missing = []
            
            if prot1 not in available_proteins:
                missing.append(prot1)
            if prot2 not in available_proteins:
                missing.append(prot2)
            
            if missing:
                reason = f"Missing PDB files: {', '.join(missing)}"
                self.invalid_pairs.append((prot1, prot2, reason))
                self.logger.debug(f"Invalid pair ({prot1}, {prot2}): {reason}")
            else:
                valid_pairs.append((prot1, prot2))
                if self.has_labels:
                    valid_labels.append(self.true_labels[i])
        
        self.protein_pairs = valid_pairs
        if self.has_labels:
            self.true_labels = valid_labels
        
        if self.invalid_pairs:
            self.logger.warning(f"Removed {len(self.invalid_pairs)} invalid pairs")
    
    def _save_error_reports(self) -> None:
        """Save error reports to the err directory."""
        # Save invalid proteins
        if self.invalid_proteins:
            invalid_proteins_file = self.err_dir / "invalid_proteins.txt"
            with open(invalid_proteins_file, 'w') as f:
                f.write("# Proteins that could not be processed\n")
                f.write("# Format: protein_id\treason\n")
                for prot_id, reason in self.invalid_proteins.items():
                    f.write(f"{prot_id}\t{reason}\n")
            self.logger.info(f"Saved invalid proteins report: {invalid_proteins_file}")
        
        # Save invalid pairs
        if self.invalid_pairs:
            invalid_pairs_file = self.err_dir / "invalid_pairs.txt"
            with open(invalid_pairs_file, 'w') as f:
                f.write("# Protein pairs that could not be processed\n")
                f.write("# Format: protein1,protein2,reason\n")
                for prot1, prot2, reason in self.invalid_pairs:
                    f.write(f"{prot1},{prot2},{reason}\n")
            self.logger.info(f"Saved invalid pairs report: {invalid_pairs_file}")
        
        # Save non-standard AA report
        if self.nonstandard_aa_proteins:
            nonstandard_file = self.err_dir / "nonstandard_amino_acids.txt"
            with open(nonstandard_file, 'w') as f:
                f.write("# Proteins with non-standard amino acids (processed with mapping)\n")
                f.write("# Format: protein_id\tnon_standard_residues\n")
                for prot_id, aas in self.nonstandard_aa_proteins.items():
                    f.write(f"{prot_id}\t{','.join(aas)}\n")
            self.logger.info(f"Saved non-standard AA report: {nonstandard_file}")
    
    def _generate_fasta(self) -> Path:
        """Generate FASTA file from extracted sequences."""
        if self.genome_mode:
            unique_proteins = set(self.protein_sequences.keys())
        else:
            unique_proteins = set()
            for prot1, prot2 in self.protein_pairs:
                unique_proteins.add(prot1)
                unique_proteins.add(prot2)

        fasta_file = self.output_dir / "proteins.fasta"

        with open(fasta_file, 'w') as f:
            for prot_id in sorted(unique_proteins):
                if prot_id in self.protein_sequences:
                    f.write(f">{prot_id}\n")
                    f.write(f"{self.protein_sequences[prot_id]}\n")

        self.logger.info(f"Generated FASTA file: {fasta_file}")
        self.logger.info(f"  Contains {len(unique_proteins)} protein sequences")

        return fasta_file
    
    def _check_multifeature_requirement(self) -> bool:
        """
        Check if multifeature generation is required.
        
        Returns:
            True if multifeature is needed for the model configuration
        """
        # This will be determined by the model configuration
        # For now, return True as a placeholder
        return True
    
    def get_multifeature_command(self, multifeature_dir: Optional[str] = None) -> str:
        """
        Generate the command for multifeature generation.
        
        Args:
            multifeature_dir: Output directory for multifeature (default: output_dir/mutifeature)
            
        Returns:
            Command string to run one_step_mutifeature.py
        """
        if multifeature_dir is None:
            multifeature_dir = self.output_dir / "mutifeature"
        
        fasta_file = self.output_dir / "proteins.fasta"

        if self.pdb_dir is None:
            return (
                "python mutifeature_tools/infer_netsurfp_head.py "
                f"--fasta {fasta_file} --output-dir {multifeature_dir}"
            )
        
        cmd = (
            f"python mutifeature_tools/one_step_mutifeature.py "
            f"-p {self.pdb_dir} "
            f"-f {fasta_file} "
            f"-o {multifeature_dir}"
        )
        
        return cmd
    
    def prompt_multifeature_generation(self, multifeature_dir: Optional[str] = None) -> str:
        """
        Display multifeature generation prompt and return the command.
        
        Args:
            multifeature_dir: Output directory for multifeature
            
        Returns:
            The multifeature generation command
        """
        cmd = self.get_multifeature_command(multifeature_dir)
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("MULTIFEATURE GENERATION REQUIRED")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("The model requires multimodal features (ProSST tokens, HMM features, etc.).")
        self.logger.info("Please run the following command to generate them:")
        self.logger.info("")
        self.logger.info(f"  {cmd}")
        self.logger.info("")
        self.logger.info("After generation completes, restart the prediction with:")
        self.logger.info(f"  --multifeature-dir {multifeature_dir or (self.output_dir / 'mutifeature')}")
        self.logger.info("")
        self.logger.info("=" * 70)
        
        return cmd


def extract_sequences_from_pdb_dir(pdb_dir: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    Utility function to extract sequences from all PDB files in a directory.
    
    Args:
        pdb_dir: Directory containing PDB files
        logger: Optional logger instance
        
    Returns:
        Dict mapping protein_id to sequence
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for PDB parsing")
    
    logger = logger or logging.getLogger(__name__)
    pdb_path = Path(pdb_dir)
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    
    sequences = {}
    parser = PDBParser(QUIET=True)
    
    for pdb_file in list(pdb_path.glob("*.pdb")) + list(pdb_path.glob("*.PDB")):
        protein_id = pdb_file.stem
        
        try:
            structure = parser.get_structure(protein_id, str(pdb_file))
            seq_parts = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':
                            continue
                        res_name = residue.get_resname().upper()
                        if res_name in AA_3TO1:
                            seq_parts.append(AA_3TO1[res_name])
                break
            
            if seq_parts:
                sequences[protein_id] = ''.join(seq_parts)
                
        except Exception as e:
            logger.warning(f"Failed to parse {pdb_file}: {e}")
    
    return sequences
