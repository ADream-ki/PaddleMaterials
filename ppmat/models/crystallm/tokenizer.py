# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Dict, Tuple, Union, Optional

from ppmat.models.crystallm.constants import (
    ATOMS, DIGITS, KEYWORDS, EXTENDED_KEYWORDS, SYMBOLS,
    UNK_TOKEN, TOKEN_ID_RANGES, CIF_KEYWORDS_REQUIRED, CIF_ATOM_KEYWORDS
)


class CIFTokenizer:
    """Tokenizer for CIF (Crystallographic Information File) format.
    
    This tokenizer converts CIF format strings into token sequences and vice versa.
    It handles special tokens, atoms, digits, keywords, and symbols specific to CIF format.
    """
    
    def __init__(self):
        """Initialize the CIFTokenizer.
        
        IMPORTANT: Token ordering must match PyTorch reference exactly:
        1. ATOMS
        2. DIGITS
        3. KEYWORDS + EXTENDED_KEYWORDS
        4. SYMBOLS
        5. SPACE_GROUPS (with _sg suffix)
        6. <unk> (last)
        """
        # Build token list in exact PyTorch order
        self._tokens = []
        
        # Add atoms (indices 0-85)
        self._tokens.extend(ATOMS)
        
        # Add digits (indices 86-95)
        self._tokens.extend(DIGITS)
        
        # Add keywords (indices 96-125)
        self._tokens.extend(KEYWORDS)
        self._tokens.extend(EXTENDED_KEYWORDS)
        
        # Add symbols (indices 126-138)
        self._tokens.extend(SYMBOLS)
        
        # Handle space groups - add suffix to disambiguate (indices 139-229)
        self._space_groups = self._load_space_groups()
        space_groups_sg = [sg + "_sg" for sg in self._space_groups]
        self._tokens.extend(space_groups_sg)
        
        # Add <unk> token at the end (index 230)
        self._tokens.append("<unk>")
        
        # Create escaped tokens for regex
        self._escaped_tokens = [re.escape(token) for token in self._tokens[:-1]]  # Exclude <unk>
        self._escaped_tokens.sort(key=len, reverse=True)
        
        # Create token to ID mapping
        self._token_to_id = {token: idx for idx, token in enumerate(self._tokens)}
        self._id_to_token = {idx: token for idx, token in enumerate(self._tokens)}
        
        # Map space group IDs back to original names (without _sg suffix)
        for sg in space_groups_sg:
            sg_id = self._token_to_id[sg]
            self._id_to_token[sg_id] = sg.replace("_sg", "")
    
    def _load_space_groups(self) -> List[str]:
        """Load space group symbols from file.
        
        IMPORTANT: Space groups must be loaded from spacegroups.txt to match
        PyTorch reference exactly. The file contains 227 space group symbols.
        
        Returns:
            List of space group symbols.
        """
        import os
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spacegroups_file = os.path.join(current_dir, "spacegroups.txt")
        
        if os.path.exists(spacegroups_file):
            with open(spacegroups_file, 'r', encoding='utf-8') as f:
                space_groups = [line.strip() for line in f if line.strip()]
            return space_groups
        else:
            # Fallback to common space groups if file not found
            # WARNING: This will cause vocab_size mismatch!
            import warnings
            warnings.warn(
                f"spacegroups.txt not found at {spacegroups_file}. "
                "Using fallback list which will cause vocab_size mismatch!"
            )
            space_groups = [
                "P1", "P-1", "P2", "P21", "C2", "Pm", "Pc", "Cm", "Cc",
                "P2/m", "P21/m", "C2/m", "P2/c", "P21/c", "C2/c",
                "P222", "P2221", "P21212", "P212121", "C222", "C2221",
                "P4", "P41", "P42", "P43", "I4", "I41",
                "P4/m", "P42/m", "P4/n", "P42/n", "I4/m", "I41/a",
                "P3", "P31", "P32", "R3",
                "P3m1", "P31m", "P3c1", "P31c", "R3m", "R3c",
                "P6", "P61", "P65", "P62", "P64", "P63",
                "P6/m", "P63/m", "P6/mmm", "P63/mmc",
                "P23", "F23", "I23", "P213", "I213",
                "Pm-3", "Pn-3", "Fm-3", "Fd-3", "Im-3", "Pa-3", "Ia-3",
                "Pm-3m", "Pn-3n", "Pm-3n", "Pn-3m", "Fm-3m", "Fm-3c",
                "Fd-3m", "Fd-3c", "Im-3m", "Ia-3d",
            ]
            return space_groups
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._tokens)
    
    @property
    def token_to_id(self) -> Dict[str, int]:
        """Get token to ID mapping."""
        return dict(self._token_to_id)
    
    @property
    def id_to_token(self) -> Dict[int, str]:
        """Get ID to token mapping."""
        return dict(self._id_to_token)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Encode a list of tokens to token IDs.
        
        Args:
            tokens: List of token strings.
        
        Returns:
            List of token IDs.
        """
        return [self._token_to_id.get(t, self._token_to_id["<unk>"]) for t in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to string.
        
        Args:
            token_ids: List of token IDs.
        
        Returns:
            Decoded string.
        """
        return "".join([self._id_to_token.get(idx, "<unk>") for idx in token_ids])
    
    def tokenize_cif(self, cif_string: str, single_spaces: bool = True) -> List[str]:
        """Tokenize a CIF format string.
        
        Args:
            cif_string: CIF format string.
            single_spaces: Whether to replace multiple spaces with single space.
        
        Returns:
            List of tokens.
        """
        # Preprocess: replace space group names with disambiguated versions
        spacegroups = "|".join(self._space_groups)
        cif_string = re.sub(
            fr'(_symmetry_space_group_name_H-M *\b({spacegroups}))\n',
            r'\1_sg\n',
            cif_string
        )
        
        # Create regex pattern
        token_pattern = "|".join(self._escaped_tokens)
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"
        
        # Tokenize
        if single_spaces:
            cif_string = re.sub(r"[ \t]+", " ", cif_string)
        
        tokens = re.findall(full_pattern, cif_string)
        
        # Replace unknown tokens
        tokens = [t if t in self._token_to_id else "<unk>" for t in tokens]
        
        return tokens
    
    def tokenize_and_encode(self, cif_string: str) -> List[int]:
        """Tokenize and encode a CIF string in one step.
        
        Args:
            cif_string: CIF format string.
        
        Returns:
            List of token IDs.
        """
        tokens = self.tokenize_cif(cif_string)
        return self.encode(tokens)
    
    def structure_to_cif_string(self, structure: 'Structure') -> str:
        """Convert a Structure object to CIF format string.
        
        Args:
            structure: Structure object to convert.
        
        Returns:
            CIF format string.
        """
        from ppmat.models.crystallm.data_models import Structure
        
        if not isinstance(structure, Structure):
            raise TypeError(f"Expected Structure, got {type(structure)}")
        
        lines = []
        
        # Data block
        lines.append("data_structure")
        lines.append("")
        
        # Cell parameters
        lines.append(f"_cell_length_a    {structure.a:.6f}")
        lines.append(f"_cell_length_b    {structure.b:.6f}")
        lines.append(f"_cell_length_c    {structure.c:.6f}")
        lines.append(f"_cell_angle_alpha {structure.alpha:.6f}")
        lines.append(f"_cell_angle_beta  {structure.beta:.6f}")
        lines.append(f"_cell_angle_gamma {structure.gamma:.6f}")
        
        if structure.volume is not None:
            lines.append(f"_cell_volume      {structure.volume:.6f}")
        
        if structure.space_group is not None:
            lines.append(f"_symmetry_space_group_name_H-M '{structure.space_group}'")
        
        if structure.formula is not None:
            lines.append(f"_chemical_formula_sum '{structure.formula}'")
        
        lines.append("")
        
        # Atom site information
        if structure.atoms:
            lines.append("loop_")
            lines.append("_atom_site_label")
            lines.append("_atom_site_type_symbol")
            lines.append("_atom_site_fract_x")
            lines.append("_atom_site_fract_y")
            lines.append("_atom_site_fract_z")
            lines.append("_atom_site_occupancy")
            
            for i, atom in enumerate(structure.atoms):
                label = atom.label if atom.label else f"{atom.element}{i+1}"
                lines.append(
                    f"{label:4s} {atom.element:2s} {atom.x:10.6f} "
                    f"{atom.y:10.6f} {atom.z:10.6f} {atom.occupancy:6.4f}"
                )
        
        return "\n".join(lines)
    
    def structure_to_tokens(self, structure: 'Structure') -> List[str]:
        """Convert a Structure object to token list.
        
        Args:
            structure: Structure object to convert.
        
        Returns:
            List of tokens.
        """
        cif_string = self.structure_to_cif_string(structure)
        return self.tokenize_cif(cif_string)
    
    def structure_to_token_ids(self, structure: 'Structure') -> List[int]:
        """Convert a Structure object to token IDs.
        
        Args:
            structure: Structure object to convert.
        
        Returns:
            List of token IDs.
        """
        tokens = self.structure_to_tokens(structure)
        return self.encode(tokens)
    
    def molecule_to_smiles_string(self, molecule: 'Molecule') -> str:
        """Convert a Molecule object to SMILES string.
        
        For now, returns the stored SMILES if available, otherwise generates a simple representation.
        
        Args:
            molecule: Molecule object to convert.
        
        Returns:
            SMILES string or simple representation.
        """
        from ppmat.models.crystallm.data_models import Molecule
        
        if not isinstance(molecule, Molecule):
            raise TypeError(f"Expected Molecule, got {type(molecule)}")
        
        if molecule.smiles:
            return molecule.smiles
        
        # Generate simple representation if SMILES not available
        composition = molecule.get_atom_composition()
        smiles_parts = []
        for element in sorted(composition.keys()):
            count = composition[element]
            if count == 1:
                smiles_parts.append(element)
            else:
                smiles_parts.append(f"{element}{count}")
        
        return "".join(smiles_parts)
    
    def molecule_to_tokens(self, molecule: 'Molecule') -> List[str]:
        """Convert a Molecule object to token list.
        
        Args:
            molecule: Molecule object to convert.
        
        Returns:
            List of tokens.
        """
        smiles = self.molecule_to_smiles_string(molecule)
        # Tokenize SMILES string
        tokens = []
        i = 0
        while i < len(smiles):
            # Try to match multi-character tokens first
            matched = False
            for token in sorted(self._tokens, key=len, reverse=True):
                if smiles[i:].startswith(token):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            
            if not matched:
                # Single character
                char = smiles[i]
                if char in self._token_to_id:
                    tokens.append(char)
                else:
                    tokens.append("<unk>")
                i += 1
        
        return tokens
    
    def molecule_to_token_ids(self, molecule: 'Molecule') -> List[int]:
        """Convert a Molecule object to token IDs.
        
        Args:
            molecule: Molecule object to convert.
        
        Returns:
            List of token IDs.
        """
        tokens = self.molecule_to_tokens(molecule)
        return self.encode(tokens)
