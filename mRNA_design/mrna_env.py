from torchgfn import Env
import torchgfn
import gfn
from gfn.env import Env, DiscreteEnv 
from typing import List, Dict
import torch 

AMINO_ACIDS: List[str] = [
    "A",  # Alanine
    "C",  # Cysteine
    "D",  # Aspartic acid
    "E",  # Glutamic acid
    "F",  # Phenylalanine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "K",  # Lysine
    "L",  # Leucine
    "M",  # Methionine
    "N",  # Asparagine
    "P",  # Proline
    "Q",  # Glutamine
    "R",  # Arginine
    "S",  # Serine
    "T",  # Threonine
    "V",  # Valine
    "W",  # Tryptophan
    "Y",  # Tyrosine
]

STOP_CODONS: List[str] = ["UAA", "UAG", "UGA"]
CODON_TABLE : Dict[str, List[str]] = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'C': ['UGU', 'UGC'],
    'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['UUU', 'UUC'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'M': ['AUG'],
    'N': ['AAU', 'AAC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    '*': ['UAA', 'UAG', 'UGA'],  # Stop codons
}

# Dictionary ambiguous amino acids to standard amino acids
AMBIGUOUS_AMINOACID_MAP: Dict[str, list[str]] = {
    "B": ["N", "D"],  # Asparagine (N) or Aspartic acid (D)
    "Z": ["Q", "E"],  # Glutamine (Q) or Glutamic acid (E)
    "X": ["A"],  # Any amino acid (typically replaced with Alanine)
    "J": ["L", "I"],  # Leucine (L) or Isoleucine (I)
    "U": ["C"],  # Selenocysteine (typically replaced with Cysteine)
    "O": ["K"],  # Pyrrolysine (typically replaced with Lysine)
}

AA_LIST = list(CODON_TABLE.keys())
AMBIG_AA_LIST = list(AMBIGUOUS_AMINOACID_MAP.keys())
CODON_MAP = {codon: i for i, codon in enumerate(sorted(set(c for codons in CODON_TABLE.values() for c in codons)))}

# Flatten and index all codons
ALL_CODONS = sorted({codon for codons in CODON_TABLE.values() for codon in codons})
CODON_TO_IDX = {codon: idx for idx, codon in enumerate(ALL_CODONS)}
IDX_TO_CODON = {idx: codon for codon, idx in CODON_TO_IDX.items()}

def protein_to_tensor(protein):

    amino_acid_counts = [0] * len(protein)
    if protein is None or protein == '':
        return torch.tensor(amino_acid_counts, dtype=torch.float)
    
    for amino_acid in protein:
        if amino_acid in AA_LIST:
            idx = AA_LIST.index(amino_acid)
            amino_acid_counts[idx] += 1

    return torch.tensor(amino_acid_counts, dtype=torch.float)

def get_synonymous_indices(amino_acid: str) -> List[int]:
    """
    Return the list of global codon indices that encode the given amino acid.
    """
    codons = CODON_TABLE.get(amino_acid, [])
    return [CODON_TO_IDX[c] for c in codons]

def get_synonymous_codons(amino_acid):

    if amino_acid in AA_LIST:
        syn = CODON_TABLE[amino_acid]
        return syn
    if amino_acid in AMBIG_AA_LIST:
        aa =  AMBIGUOUS_AMINOACID_MAP[amino_acid][0]   # we will just take the first amino acid
        syn = CODON_TABLE[aa]
    return syn

def compute_gc_content_from_indices(indices: torch.LongTensor) -> torch.FloatTensor:
    """
    Given a tensor of codon indices (batch x seq_len), compute GC-content (%) per sequence.
    """
    # Expand to list of strings
    contents = []
    for seq in indices:
        rna = ''.join(IDX_TO_CODON[int(i)] for i in seq) # mRNA sequence
        gc = (rna.count('G') + rna.count('C')) / len(rna) * 100 if len(rna) > 0 else 0.0
        contents.append(gc)
    return torch.tensor(contents, dtype=torch.float, device=indices.device)

class CodonDesignEnv(DiscreteEnv):
    """
    Environment for designing mRNA codon sequences for a given protein. States are LongTensors of shape (batch, t) representing chosen codon indices.
    Action space is global codon set of size len(ALL_CODONS); dynamic masks restrict to synonymous codons at each step.
    Rewards are GC-content of the full sequence so far.
    """

    def __init__(
        self,
        protein_seq: str,
        dummy_action=None, exit_action=None, sf=None
    ):
        self.protein_seq = protein_seq
        self.seq_length = len(protein_seq)
        self.n_actions = len(ALL_CODONS)
        self.device = torch.device('cpu')

        # Precompute valid indices per position
        self.syn_indices = [get_synonymous_indices(aa) for aa in protein_seq]

        # Initial empty state
        initial_state = torch.empty((0,), dtype=torch.long, device=self.device)

        super().__init__(
            n_actions=self.n_actions,
            s0=initial_state,  
            state_shape=(None,),    # variable-length vector of codon indices
            action_shape=(),                  
            dummy_action=dummy_action,
            exit_action=exit_action,
            sf=sf
        )
 
    def step(
        self,
        states: torch.LongTensor,
        actions: torch.LongTensor,
    ) -> torch.LongTensor:
        # Append action indices to states
        # states: (batch, t), actions: (batch,)
        actions = actions.unsqueeze(-1)
        return torch.cat([states, actions], dim=1)
    
    def backward_step(
        self,
        states: torch.LongTensor,
    ) -> torch.LongTensor:
        # Remove last codon index
        return states[:, :-1]
    
    def update_masks(
        self,
        states: torch.LongTensor,
    ) -> torch.BoolTensor:
        
        # For each sequence in batch, mask only synonymous codons for next aa.
        batch = states.shape[0]
        next_pos = states.shape[1]

        if next_pos >= self.seq_length:
            # No valid actions beyond terminal
            return torch.zeros((batch, self.n_actions), dtype=torch.bool, device=self.device)
        
        valid = torch.zeros((batch, self.n_actions), dtype=torch.bool, device=self.device)
        valid_indices = self.syn_indices[next_pos]
        valid[:, valid_indices] = True
        return valid

    def reward(
        self,
        states: torch.LongTensor,
    ) -> torch.FloatTensor:
        # Compute GC-content percentage of each sequence
        return compute_gc_content_from_indices(states)
    
    def reset(self, batch_size: int) -> torch.LongTensor:
        # Return batch of empty sequences
        return torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
    
    def is_terminal(
        self,
        states: torch.LongTensor,
    ) -> torch.BoolTensor:
        # Terminal when sequence length equals protein length
        return states.shape[1] >= self.seq_length




















