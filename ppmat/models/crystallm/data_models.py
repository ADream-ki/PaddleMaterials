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

"""Data models for CrystaLLM: Structure, Molecule, and Atom classes."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from ppmat.models.crystallm.constants import (
    EPSILON, COORDINATE_TOLERANCE, LATTICE_PARAM_MIN, LATTICE_PARAM_MAX,
    ANGLE_MIN, ANGLE_MAX
)


@dataclass
class Atom:
    """Represents a single atom in a crystal structure or molecule.
    
    Attributes:
        element (str): Chemical element symbol (e.g., 'Si', 'O', 'C').
        x (float): Fractional or Cartesian x-coordinate.
        y (float): Fractional or Cartesian y-coordinate.
        z (float): Fractional or Cartesian z-coordinate.
        charge (float, optional): Partial charge on the atom. Defaults to 0.0.
        occupancy (float, optional): Occupancy factor (0-1). Defaults to 1.0.
        label (str, optional): Atom label/identifier. Defaults to None.
        is_fractional (bool, optional): Whether coordinates are fractional. Defaults to True.
    """
    
    element: str
    x: float
    y: float
    z: float
    charge: float = 0.0
    occupancy: float = 1.0
    label: Optional[str] = None
    is_fractional: bool = True
    
    def __post_init__(self):
        """Validate atom parameters."""
        if not isinstance(self.element, str) or len(self.element) == 0:
            raise ValueError(f"Invalid element: {self.element}")
        
        if not (0.0 <= self.occupancy <= 1.0):
            raise ValueError(f"Occupancy must be between 0 and 1, got {self.occupancy}")
        
        # Validate coordinates are within reasonable bounds
        if self.is_fractional:
            # Fractional coordinates should be roughly in [0, 1], but allow some tolerance
            for coord, name in [(self.x, 'x'), (self.y, 'y'), (self.z, 'z')]:
                if not (-1.0 - COORDINATE_TOLERANCE <= coord <= 2.0 + COORDINATE_TOLERANCE):
                    raise ValueError(f"Fractional coordinate {name}={coord} out of bounds")
    
    def get_coordinates(self) -> Tuple[float, float, float]:
        """Get atom coordinates as a tuple."""
        return (self.x, self.y, self.z)
    
    def to_dict(self) -> Dict:
        """Convert atom to dictionary."""
        return {
            'element': self.element,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'charge': self.charge,
            'occupancy': self.occupancy,
            'label': self.label,
            'is_fractional': self.is_fractional,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Atom':
        """Create Atom from dictionary."""
        return cls(**data)


@dataclass
class Structure:
    """Represents a crystal structure with lattice parameters and atomic positions.
    
    Attributes:
        a (float): Lattice parameter a (in Angstroms).
        b (float): Lattice parameter b (in Angstroms).
        c (float): Lattice parameter c (in Angstroms).
        alpha (float): Lattice angle alpha (in degrees).
        beta (float): Lattice angle beta (in degrees).
        gamma (float): Lattice angle gamma (in degrees).
        atoms (List[Atom]): List of atoms in the structure.
        space_group (str, optional): Space group symbol. Defaults to None.
        formula (str, optional): Chemical formula. Defaults to None.
        volume (float, optional): Unit cell volume. Defaults to None.
    """
    
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    atoms: List[Atom] = field(default_factory=list)
    space_group: Optional[str] = None
    formula: Optional[str] = None
    volume: Optional[float] = None
    
    def __post_init__(self):
        """Validate structure parameters."""
        # Validate lattice parameters
        for param, name in [(self.a, 'a'), (self.b, 'b'), (self.c, 'c')]:
            if not (LATTICE_PARAM_MIN <= param <= LATTICE_PARAM_MAX):
                raise ValueError(
                    f"Lattice parameter {name}={param} out of bounds "
                    f"[{LATTICE_PARAM_MIN}, {LATTICE_PARAM_MAX}]"
                )
        
        # Validate angles
        for angle, name in [(self.alpha, 'alpha'), (self.beta, 'beta'), (self.gamma, 'gamma')]:
            if not (ANGLE_MIN <= angle <= ANGLE_MAX):
                raise ValueError(
                    f"Lattice angle {name}={angle} out of bounds "
                    f"[{ANGLE_MIN}, {ANGLE_MAX}]"
                )
        
        # Validate atoms
        if not isinstance(self.atoms, list):
            raise ValueError("atoms must be a list")
        
        for atom in self.atoms:
            if not isinstance(atom, Atom):
                raise ValueError(f"All atoms must be Atom instances, got {type(atom)}")
    
    def get_lattice_parameters(self) -> Tuple[float, float, float, float, float, float]:
        """Get lattice parameters as a tuple."""
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
    
    def get_num_atoms(self) -> int:
        """Get number of atoms in the structure."""
        return len(self.atoms)
    
    def get_atom_types(self) -> List[str]:
        """Get list of unique atom types."""
        return sorted(list(set(atom.element for atom in self.atoms)))
    
    def get_atom_composition(self) -> Dict[str, int]:
        """Get atom composition (element -> count)."""
        composition = {}
        for atom in self.atoms:
            composition[atom.element] = composition.get(atom.element, 0) + 1
        return composition
    
    def to_dict(self) -> Dict:
        """Convert structure to dictionary."""
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'atoms': [atom.to_dict() for atom in self.atoms],
            'space_group': self.space_group,
            'formula': self.formula,
            'volume': self.volume,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Structure':
        """Create Structure from dictionary."""
        atoms_data = data.pop('atoms', [])
        atoms = [Atom.from_dict(atom_data) for atom_data in atoms_data]
        return cls(atoms=atoms, **data)


@dataclass
class Molecule:
    """Represents a molecule with atoms and bonds.
    
    Attributes:
        atoms (List[Atom]): List of atoms in the molecule.
        bonds (List[Tuple[int, int, int]], optional): List of bonds as (atom_idx1, atom_idx2, bond_order).
            Defaults to empty list.
        formula (str, optional): Chemical formula. Defaults to None.
        smiles (str, optional): SMILES representation. Defaults to None.
        properties (Dict, optional): Additional molecular properties. Defaults to empty dict.
    """
    
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Tuple[int, int, int]] = field(default_factory=list)
    formula: Optional[str] = None
    smiles: Optional[str] = None
    properties: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate molecule parameters."""
        if not isinstance(self.atoms, list):
            raise ValueError("atoms must be a list")
        
        for atom in self.atoms:
            if not isinstance(atom, Atom):
                raise ValueError(f"All atoms must be Atom instances, got {type(atom)}")
        
        if not isinstance(self.bonds, list):
            raise ValueError("bonds must be a list")
        
        # Validate bonds
        num_atoms = len(self.atoms)
        for bond in self.bonds:
            if not isinstance(bond, (tuple, list)) or len(bond) != 3:
                raise ValueError(f"Each bond must be a tuple of (idx1, idx2, order), got {bond}")
            
            idx1, idx2, order = bond
            if not (0 <= idx1 < num_atoms and 0 <= idx2 < num_atoms):
                raise ValueError(
                    f"Bond indices {idx1}, {idx2} out of range [0, {num_atoms-1}]"
                )
            
            if order not in [1, 2, 3, 1.5]:  # Single, double, triple, aromatic
                raise ValueError(f"Invalid bond order: {order}")
    
    def get_num_atoms(self) -> int:
        """Get number of atoms in the molecule."""
        return len(self.atoms)
    
    def get_num_bonds(self) -> int:
        """Get number of bonds in the molecule."""
        return len(self.bonds)
    
    def get_atom_types(self) -> List[str]:
        """Get list of unique atom types."""
        return sorted(list(set(atom.element for atom in self.atoms)))
    
    def get_atom_composition(self) -> Dict[str, int]:
        """Get atom composition (element -> count)."""
        composition = {}
        for atom in self.atoms:
            composition[atom.element] = composition.get(atom.element, 0) + 1
        return composition
    
    def get_neighbors(self, atom_idx: int) -> List[int]:
        """Get neighbor atom indices for a given atom."""
        neighbors = []
        for idx1, idx2, _ in self.bonds:
            if idx1 == atom_idx:
                neighbors.append(idx2)
            elif idx2 == atom_idx:
                neighbors.append(idx1)
        return neighbors
    
    def to_dict(self) -> Dict:
        """Convert molecule to dictionary."""
        return {
            'atoms': [atom.to_dict() for atom in self.atoms],
            'bonds': self.bonds,
            'formula': self.formula,
            'smiles': self.smiles,
            'properties': dict(self.properties),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Molecule':
        """Create Molecule from dictionary."""
        atoms_data = data.pop('atoms', [])
        atoms = [Atom.from_dict(atom_data) for atom_data in atoms_data]
        return cls(atoms=atoms, **data)
