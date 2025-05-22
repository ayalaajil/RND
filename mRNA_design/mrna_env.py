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
IDX_TO_CODON = {v: k for k, v in CODON_MAP.items()}

def protein_to_tensor(protein):

    amino_acid_counts = [0] * len(protein)
    if protein is None or protein == '':
        return torch.tensor(amino_acid_counts, dtype=torch.float)
    
    for amino_acid in protein:
        if amino_acid in AA_LIST:
            idx = AA_LIST.index(amino_acid)
            amino_acid_counts[idx] += 1

    return torch.tensor(amino_acid_counts, dtype=torch.float)


def get_synonymous_codons(amino_acid):

    if amino_acid in AA_LIST:
        syn = CODON_TABLE[amino_acid]
        return syn
    if amino_acid in AMBIG_AA_LIST:
        aa =  AMBIGUOUS_AMINOACID_MAP[amino_acid][0]   # we will just take the first amino acid
        syn = CODON_TABLE[aa]
    return syn




class CodonDesignEnv(DiscreteEnv):

    def __init__(self, protein_seq, dummy_action=None, exit_action=None, sf=None):
        
        self.protein = protein_seq
        self.seq_length = len(self.protein)

        self.codon_choices = [get_synonymous_codons(aa) for aa in protein_seq]
        self.action_space_sizes = [len(choices) for choices in self.codon_choices]

        n_actions = 64 # Total number of codons in the codon table

        device = torch.device("cpu")

        s0 = torch.zeros(0, dtype=torch.float, device=device) # Start with an empty sequence
        self.state_shape = (self.seq_length,)  # Number of amino acids <=> Codons

        super().__init__(n_actions, 
                         s0, 
                         state_shape=self.state_shape, 
                         action_shape=(1,),  # Each action is 1 codon,
                         dummy_action= dummy_action,
                         exit_action=exit_action, 
                         sf=sf)


    def step(self, state, action): 

        # state is the mRNA sequence, action is the chosen codon
        # Append the chosen codon to the mRNA sequence
        return torch.cat([state, action.unsqueeze(-1)], dim=-1)
    
    def reset(self, batch_size):
        return ["" for _ in range(batch_size)]  # initial empty sequences
    
    def is_terminal(self, states):
        return [len(s) // 3 >= len(self.protein) for s in states]


































class CodonDesignEnv(Env):
    def __init__(self, protein_seq, codon_table):
        self.protein = protein_seq
        self.codon_choices = [get_synonymous_codons(aa) for aa in protein_seq]
        self.num_steps = len(protein_seq)

    def reset(self, batch_size):
        return ["" for _ in range(batch_size)]  # initial empty sequences

    def step(self, states, actions):
        # Append the chosen codon to each sequence
        new_states = []
        for s, a in zip(states, actions):
            codon = self.codon_choices[len(s) // 3][a]  # a is index into allowed codons
            new_states.append(s + codon)
        return new_states

    def is_terminal(self, states):
        return [len(s) // 3 >= len(self.protein) for s in states]
