from ccdc import io
from ccdc.io import CrystalWriter
from pymatgen.symmetry.groups import sg_symbol_from_int_number, SpaceGroup
from pymatgen.core import Structure, Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.io.cif import CifWriter
import argparse
import os
import uuid
import warnings
import numpy as np

script_path, script_name = os.path.split(os.path.realpath(__file__))
temp_cif_path = "/tmp/temp-{}.cif".format(uuid.uuid4().hex)

def get_block(cif_lines, keyword, start):

    """
    Separate a cif into blocks based on "loop_" sections.

        Parameters:
            cif_lines (list of str): list of lines in cif file
            keyword (str): keyword identifying block (e.g., "atom" for atoms
                       "symmetry" for symmetry operations, "geom" for geometry
                       (bonding) information, etc.)
            start (int): the line number of beginning of block (line after "loop_")
    
        Returns:
            block (list of str): list of lines in the "block" from the cif
    """

    for num, line in enumerate(cif_lines[start:]):
        if line[0] == "_" and line.strip("\n").split("_")[1] != keyword:
            end = num + start
            break
    try:
        block = cif_lines[start:end]
    except UnboundLocalError:
        block = cif_lines[start:]
    # Remove any "loop_" lines
    block = [val for val in block if val != "loop_\n"]

    return block

def get_dict(lst, good_atoms, label_key):

    """
    Get a dictionary of structural attributes of a cif containing only certain
    atoms. For example, to get a dictionary of atoms and their coordinates for only
    a certain list of atoms. This works given a list of lines from a "block" in a cif
    (identified by the "loop_" sections)

        Parameters:
            lst (list of str): List of lines to consider for dictionary
            good_atoms (list of str): Atom labels to consider for dictionary
            label_key (str): the header which will be used as the keys of the dict
                             (e.g., for atom labels use _atom_site_label)
        
        Returns:
            dict (dict): Dictionary of specified keys and attributes (properties)
                         (properties are the remaining headers in the "loop_" section)
    """

    dict = {}
    for num, line in enumerate(lst):
        if line[0] == "_":
            dict[line.strip().strip("\n")] = []
        else:
            line = line.strip().replace("(", "").replace(")", "").split()
            if line[list(dict.keys()).index(label_key)] in good_atoms:
                for n, prop in enumerate(dict.keys()):
                    dict[prop].append(line[n])
        
    return dict

def get_asymmetric_unit(ref):

    """ 
    Retrieve a cif from the CSD and return the
    asymmetric unit molecule atoms. Also, write the crystal to a 
    temporary cif to get atomic coordinates of the crystal
    (no way to do this with CSD).
        
        Parameters:
            ref (str): the CSD refcode for the structure
        Returns:
            atoms (list of str): the atom labels of the asymmetric unit molecule
        
    """

    csd_reader = io.EntryReader('CSD')
    cryst = csd_reader.entry(ref).crystal
    cryst.centre_molecule()
    mol = cryst.asymmetric_unit_molecule
    atoms = [str(atom).strip("Atom(").strip(")") for atom in mol.atoms]
    with CrystalWriter(temp_cif_path) as mol_writer:
        mol_writer.write(cryst)

    return atoms

def convert_to_p1(ops, asym_unit_atoms, asym_unit_coords):

    """
    From a list of symmetry operations in string format and a list of atoms and
    their coordinates, generate a full unit cell in P1 symmetry.

        Parameters:
            ops (list of str): list
            asym_unit_atoms (list of str): list of elements in asymmetric unit
            asym_unit_coords (list of float): list of fractional coordinates of
                                    atoms in asymmetric unit
        
        Returns:
            new_species (list of str): species in the full unit cell
            new_coords (numpy array): array of fractional coordinates
                                        of atoms in full unit cell

    """

    op_matrices = []
    unit_cell_atoms = []
    unit_cell_coords = []
    # Need to apply symmetry operations specified in cif manually,
    # space group isn't good enough
    a = np.zeros((4, 4), dtype=np.float32)
    # Need to instantiate SymmOp class with some 4x4 affine matrix to
    # use from_xyz_string function
    sym_op_obj = SymmOp(a)
    for op in ops:
        op_matrices.append(sym_op_obj.from_xyz_string(op).affine_matrix)

    for num, atom in enumerate(asym_unit_atoms):
        for op in op_matrices:
            unit_cell_coords.append(op @ asym_unit_coords[num])
            unit_cell_atoms.append(atom)

    unit_cell_coords = np.asarray(unit_cell_coords)
    unit_cell_coords = unit_cell_coords[:, :-1]

    return unit_cell_atoms, unit_cell_coords

def remove_duplicate_atoms(pm_struct):
    """
    Remove atoms which have identical fractional coordinates.

        Parameters:
            pm_struct (Pymatgen Structure object): structure from which
                                                   to remove duplicates
        Returns:
            pm_struct (Pymatgen Structure object): structure with duplicates
                                                   removed
    """

    coords_check = []
    bad_indices = []
    for num, atom in enumerate(pm_struct):
        for coord in coords_check:
            if all(coord == atom.frac_coords):
                bad_indices.append(num)
        else:
            coords_check.append(atom.frac_coords)

    pm_struct.remove_sites(bad_indices)

    return pm_struct

def csd_to_pymatgen(path, atoms):
    """
    Manually parse a CSD cif to generate a pymatgen structure which contains
    only a list of atoms specified (in this case, the asymmetric unit). This
    function returns a Pymatgen structure which is in P1 symmetry by applying
    the symmetry operations found in the cif.

        Parameters:
            path (str): path to the cif file
            atoms (list): list of the atoms in the asymmetric unit
    
        Returns:
            struct (Pymatgen Structure object): Structure in P1 symmetry
    """
    with open(path) as f:
        cif = f.readlines()
        f.close()

    blocks = {"symmetry": None,
            "atom": None, 
            "geom": None}

    for num, line in enumerate(cif):
        if "loop_" in line:
            for block in blocks.keys():
                if cif[num + 1].strip().split("_")[1] == block:
                    blocks[block] = get_block(cif, block, num)
        elif "_symmetry_space_group_name_H-M" in line:
            space_group_name_HM = line.strip().split()[-1]
        elif "_symmetry_Int_Tables_number" in line:
            int_tables_num = line.strip().split()[-1]
        elif "_space_group_name_Hall" in line:
            space_group_name_hall = line.strip().split()[-1]
        elif "cell_length_a" in line:
            cell_length_a = float(line.strip().
                                  replace("(", "").
                                  replace(")", "").
                                  split()[-1])
        elif "cell_length_b" in line:
            cell_length_b = float(line.strip().
                                  replace("(", "").
                                  replace(")", "").
                                  split()[-1])
        elif "cell_length_c" in line:
            cell_length_c = float(line.strip().
                                  replace("(", "").
                                  replace(")", "").
                                  split()[-1])
        elif "cell_angle_alpha" in line:
            cell_angle_alpha = float(line.strip().
                                     replace("(", "").
                                     replace(")", "").
                                     split()[-1])
        elif "cell_angle_beta" in line:
            cell_angle_beta = float(line.strip().
                                         replace("(", "").
                                         replace(")", "").
                                         split()[-1])
        elif "cell_angle_gamma" in line:
            cell_angle_gamma = float(line.strip().
                                          replace("(", "").
                                          replace(")", "").
                                          split()[-1])

    num_cols = 0
    atom_dict = {}
    coords = []

    atom_dict = get_dict(blocks["atom"], atoms, "_atom_site_label")
    for x, y, z in zip(atom_dict["_atom_site_fract_x"],
                       atom_dict["_atom_site_fract_y"],
                       atom_dict["_atom_site_fract_z"]):
        coords.append([float(x), float(y), float(z), 1])

    species=atom_dict["_atom_site_type_symbol"]
    sg=SpaceGroup.from_int_number(int(int_tables_num))
    # Get only the operations (not headers) from the Symmetry block
    sym_ops = [val for val in blocks["symmetry"] if val[0] != "_"]
    # Get rid of any numbering "e.g., 1 x,y,z" from the symmetry operations
    for i in range(len(sym_ops)):
        sym_ops[i] = [val.split()[-1] for val in sym_ops[i].strip().split(',')]
        sym_ops[i] = ",".join(sym_ops[i])

    species, coords = convert_to_p1(sym_ops, species, coords)

    lattice = Lattice.from_parameters(a=cell_length_a, b=cell_length_b,
                                      c=cell_length_c, alpha=cell_angle_alpha,
                                      beta=cell_angle_beta,
                                      gamma=cell_angle_gamma)
    struct = Structure(lattice=lattice, species=species,
                       coords=coords,to_unit_cell=True)
    # Need to remove duplicates since some atoms lie on
    # boundaries when converted to P1
    struct = remove_duplicate_atoms(struct)

    return struct

def main(path):

    """
    Get the asymmetric unit of a CSD structure and convert it to a Pymatgen
    Structure object in P1 symmetry. Then, write the cif to a specified path.

        Parameters:
            path (str): Path to write the cif
        
        Returns:
            mof_p1 (Pymatgen Structure object): MOF in P1 symmetry
    """

    ref = path.split('/')[-1].split("_P1.cif")[0]
    asymmetric_unit_atoms = get_asymmetric_unit(ref)
    mof_p1 = csd_to_pymatgen(temp_cif_path, asymmetric_unit_atoms)
    os.remove(temp_cif_path)
    CifWriter(mof_p1).write_file(path)
    
    return mof_p1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description = "Retrieve crystal structure from the CSD," +
                  "and convert it to P1 symmetry.")
    parser.add_argument("refcode", metavar="r", type=str,
                        help="CSD Refcode of the MOF")
    parser.add_argument("--write_dir", type=str,
                        default=os.getcwd(),
                        help="Directory to write the cif." + 
                        " Defaults to current working directory.",)
    args = parser.parse_args()
    refcode = args.refcode
    write_dir = args.write_dir
    final_cif_path = "{}/{}_P1.cif".format(write_dir, refcode)

    main(final_cif_path)
