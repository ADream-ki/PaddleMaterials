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

"""Constants for CrystaLLM model."""

# Token types
ATOMS = [
    "Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
    "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
    "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
    "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
    "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"
]

DIGITS = [str(d) for d in range(10)]

KEYWORDS = [
    "_cell_length_b",
    "_atom_site_occupancy",
    "_atom_site_attached_hydrogens",
    "_cell_length_a",
    "_cell_angle_beta",
    "_symmetry_equiv_pos_as_xyz",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_structural",
    "_chemical_name_systematic",
    "_atom_site_fract_y",
    "_atom_site_symmetry_multiplicity",
    "_chemical_formula_sum",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_cell_length_c",
    "_atom_site_B_iso_or_equiv",
    "_symmetry_equiv_pos_site_id",
    "_cell_volume",
    "_atom_site_fract_z",
    "_cell_angle_alpha",
    "_cell_formula_units_Z",
    "loop_",
    "data_"
]

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",
    "_atom_type_radius",
    "_atom_type_ionic_radius",
    "_atom_type_oxidation_number"
]

SYMBOLS = ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

# Unknown token (added at the end of vocabulary)
UNK_TOKEN = "<unk>"

# Token ID ranges (matching PyTorch reference)
# Note: Space groups count will be determined at runtime
TOKEN_ID_RANGES = {
    "atoms": (0, len(ATOMS)),
    "digits": (len(ATOMS), len(ATOMS) + len(DIGITS)),
    "keywords": (len(ATOMS) + len(DIGITS), 
                 len(ATOMS) + len(DIGITS) + len(KEYWORDS) + len(EXTENDED_KEYWORDS)),
    "symbols": (len(ATOMS) + len(DIGITS) + len(KEYWORDS) + len(EXTENDED_KEYWORDS),
                len(ATOMS) + len(DIGITS) + len(KEYWORDS) + len(EXTENDED_KEYWORDS) + len(SYMBOLS)),
}

# Total vocabulary size (without space groups, will be ~231 with 91 space groups)
BASE_VOCAB_SIZE = len(ATOMS) + len(DIGITS) + len(KEYWORDS) + len(EXTENDED_KEYWORDS) + len(SYMBOLS) + 1  # +1 for <unk>

# CIF format constants
CIF_KEYWORDS_REQUIRED = [
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
]

CIF_ATOM_KEYWORDS = [
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
]

# Numerical constants
EPSILON = 1e-5
COORDINATE_TOLERANCE = 1e-6
LATTICE_PARAM_MIN = 0.1
LATTICE_PARAM_MAX = 100.0
ANGLE_MIN = 1.0
ANGLE_MAX = 179.0
