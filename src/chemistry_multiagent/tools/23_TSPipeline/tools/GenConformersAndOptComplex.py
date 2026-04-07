import random
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rmsd import kabsch_rmsd, reorder_inertia_hungarian, centroid, int_atom
import os

# ------------------------------
# 核心工具函数（复用原逻辑）
# ------------------------------
def calculate_rmsd_directly(atoms1, coords1, atoms2, coords2, reorder=True, no_hydrogen=False):
    '''
    Description: Computes the Root Mean Square Deviation (RMSD) between two sets of atomic coordinates, optionally filtering hydrogens, reordering atoms for alignment using Hungarian algorithm on atomic numbers, centering coordinates via centroids, and applying Kabsch algorithm for superposition. Raises ValueError if atom counts mismatch after filtering.
    Parameters:
        atoms1 (list of str): Atomic symbols of the first molecule.
        coords1 (numpy.ndarray): Coordinates of the first molecule.
        atoms2 (list of str): Atomic symbols of the second molecule.
        coords2 (numpy.ndarray): Coordinates of the second molecule.
        reorder (bool, optional): Whether to reorder atoms for alignment. Defaults to True.
        no_hydrogen (bool, optional): Whether to filter out hydrogen atoms. Defaults to False.
    
    Output: float: The RMSD between the two sets of coordinates.
    '''
    # 过滤氢原子（如果需要）
    if no_hydrogen:
        # 保留非氢原子的索引（H的原子符号为'H'）
        mask1 = [i for i, atom in enumerate(atoms1) if atom != 'H']
        mask2 = [i for i, atom in enumerate(atoms2) if atom != 'H']
        # 应用过滤
        atoms1 = [atoms1[i] for i in mask1]
        coords1 = coords1[mask1]
        atoms2 = [atoms2[i] for i in mask2]
        coords2 = coords2[mask2]

    # 检查过滤后原子数量是否匹配
    if len(atoms1) != len(atoms2):
        raise ValueError("原子数量不匹配，无法计算 RMSD")

    if reorder:
        atoms1_int = np.array([int_atom(atom) for atom in atoms1])
        atoms2_int = np.array([int_atom(atom) for atom in atoms2])
        # coords2 = coords2[reorder_hungarian(atoms1, atoms2, coords1, coords2)]
        coords2 = coords2[reorder_inertia_hungarian(atoms1_int, atoms2_int, coords1, coords2)]
    coords1_centered = coords1 - centroid(coords1)
    coords2_centered = coords2 - centroid(coords2)
    return kabsch_rmsd(coords1_centered, coords2_centered)


def generate_diverse_conformers(mol, rmsd_threshold=-1.0, max_conformers=None, max_attempts=1000, max_iter=200):
    '''
    Description: Generates and optimizes diverse 3D conformers for a molecule: adds hydrogens, determines number of conformers based on rotatable bonds, embeds them with optional RMSD pruning, optimizes each with MMFF force field, computes energies, sorts by energy, and selects up to max_conformers. Raises RuntimeError if no conformers are generated.

    Parameters:
        - mol: rdkit.Chem.rdchem.Mol (input molecule without hydrogens)
        - rmsd_threshold: float (default: -1.0, threshold for pruning similar conformers during embedding; negative means no pruning)
        - max_conformers: int or None (default: None, maximum number of conformers to return; None means all)
        - max_attempts: int (default: 1000, maximum number of embedding attempts, capped by formula)
        - max_iter: int (default: 200, maximum iterations for MMFF optimization)
    Output:
    tuple: (rdkit.Chem.rdchem.Mol (molecule with hydrogens and conformers), list of int (sorted conformer IDs by energy))
    '''
    mol = Chem.AddHs(mol)
    num_rot_bonds = CalcNumRotatableBonds(mol, strict=True)
    num_confs = min(10 + 4 ** num_rot_bonds, max_attempts)
    cids = EmbedMultipleConfs(mol, numConfs=num_confs, pruneRmsThresh=rmsd_threshold, randomSeed=42)
    if not cids:
        raise RuntimeError("无法生成构象")
    
    conf_energies = []
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=max_iter)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=cid)
        conf_energies.append((cid, ff.CalcEnergy()))
    
    conf_energies.sort(key=lambda x: x[1])
    sorted_cids = [cid for cid, _ in conf_energies]
    return mol, (sorted_cids[:max_conformers] if max_conformers else sorted_cids)


def read_smiles(smiles, removeHs=False):
    '''
    Description: Parses a SMILES string into an RDKit molecule object using custom parser parameters, optionally removing hydrogens. Raises ValueError if parsing fails.
    Parameters:
        - smiles: str (SMILES string to parse)
        - removeHs: bool (default: False, whether to remove hydrogens during parsing)
    Output:
    rdkit.Chem.rdchem.Mol (the parsed molecule)
    '''
    parser = Chem.SmilesParserParams()
    parser.removeHs = removeHs
    mol = Chem.MolFromSmiles(smiles, params=parser)
    if mol is None:
        raise ValueError(f"无法解析SMILES: {smiles}")
    return mol


def calculate_centroid(mol, conf_id):
    '''
    Description: Calculates the centroid (center of mass) of a molecule's atoms in a specific conformer.

    Parameters:
        - mol: rdkit.Chem.rdchem.Mol (input molecule)
        - conf_id: int (conformer ID to use for calculation)
    Output:
    numpy.ndarray (centroid coordinates)
    '''
    conf = mol.GetConformer(conf_id)
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    return np.mean([(p.x, p.y, p.z) for p in coords], axis=0)


def translate_molecule(mol, conf_id, translation):
    '''
    Description: Translates a molecule's atoms in a specific conformer by a given vector.

    Parameters:
        - mol: rdkit.Chem.rdchem.Mol (input molecule)
        - conf_id: int (conformer ID to use for translation)
        - translation: numpy.ndarray (translation vector [x, y, z])
    Output:
    rdkit.Chem.rdchem.Mol (the translated molecule)
    '''
    conf = mol.GetConformer(conf_id)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + translation[0], pos.y + translation[1], pos.z + translation[2]))
    return mol


def rotate_molecule(mol, conf_id, angle_x=0, angle_y=0, angle_z=0):
    '''
    Description: Rotates the molecule around its centroid by specified Euler angles (x, y, z) using rotation matrices, translating to/from origin for rotation, modifying the conformer in place.
    Parameters:
    - mol: rdkit.Chem.rdchem.Mol (molecule to rotate)
    - conf_id: int (ID of the conformer to modify)
    - angle_x: float (default: 0, rotation angle in degrees around x-axis)
    - angle_y: float (default: 0, rotation angle in degrees around y-axis)
    - angle_z: float (default: 0, rotation angle in degrees around z-axis)

    Output:
    rdkit.Chem.rdchem.Mol (the modified molecule)
    '''
    conf = mol.GetConformer(conf_id)
    centroid = calculate_centroid(mol, conf_id)
    translate_molecule(mol, conf_id, -centroid)
    
    rx, ry, rz = np.radians([angle_x, angle_y, angle_z])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_coords = R @ np.array([pos.x, pos.y, pos.z])
        conf.SetAtomPosition(i, (new_coords[0], new_coords[1], new_coords[2]))
    
    translate_molecule(mol, conf_id, centroid)
    return mol


def optimize_with_constraints(merged_mol, conf_id, constraints, max_iter=10000, force_tol=1e-05, energy_tol=1e-07):
    '''
    Description: The function optimize_with_constraints performs a geometry optimization of a molecule under specified distance constraints using the MMFF (Merck Molecular Force Field). It allows controlling key interatomic distances while minimizing the molecular energy, making it useful for constructing reasonable initial structures for reaction complexes, transition states, or constrained conformational studies. During the optimization, a force field is built for the molecule, user-defined distance constraints between atom pairs are applied, and the molecular geometry is minimized iteratively until either the maximum number of iterations or the convergence thresholds for forces and energy are reached.
    Parameters:
    - merged_mol: rdkit.Chem.rdchem.Mol (molecule to optimize)
    - conf_id: int (ID of the conformer to optimize)
    - constraints: list of tuple (each: (int idx1, int idx2, bool relative, float min_dist, float max_dist, float force_constant))
    - max_iter: int (default: 10000, maximum minimization iterations)
    - force_tol: float (default: 1e-05, force tolerance for convergence)
    - energy_tol: float (default: 1e-07, energy tolerance for convergence)

    Output:
    merged_mol: rdkit.Chem.rdchem.Mol (the optimized molecule), The molecule object after constrained optimization. The coordinates of the specified conformer are updated.
    conf_id: int (the same conformer ID passed in)
    '''
    mmff_props = AllChem.MMFFGetMoleculeProperties(merged_mol)
    ff = AllChem.MMFFGetMoleculeForceField(merged_mol, mmff_props, confId=conf_id, ignoreInterfragInteractions=False)
    for (idx1, idx2, relative, min_dist, max_dist, force_constant) in constraints:
        ff.MMFFAddDistanceConstraint(idx1, idx2, relative, min_dist, max_dist, force_constant)
    ff.Minimize(maxIts=max_iter, forceTol=force_tol, energyTol=energy_tol)
    return merged_mol, conf_id


def write_xyz_file(elements, coordinates, conf_id, filename, charge=0, multiplicity=1):
    '''
    Description: Writes atomic elements and coordinates to an XYZ file, including atom count, comment with conf_id, charge, and multiplicity.
    Parameters:
    - elements: list or numpy.ndarray (length: n_atoms, dtype: str, atomic symbols)
    - coordinates: list or numpy.ndarray (shape: (n_atoms, 3), dtype: float, atomic positions)
    - conf_id: int (conformer ID for comment)
    - filename: str (base filename without .xyz extension)
    - charge: int (default: 0, total molecular charge)
    - multiplicity: int (default: 1, spin multiplicity)
    Output:
    None
    '''
    with open(f"{filename}.xyz", 'w') as f:
        f.write(f"{len(elements)}\n")
        f.write(f"Atoms: {len(elements)}, ConfID: {conf_id} {charge} {multiplicity}\n")
        for elem, coord in zip(elements, coordinates):
            f.write(f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


# ------------------------------
# 新增拆分的功能函数
# ------------------------------
def prepare_molecules_and_conformers(smiles_list, conformer_rmsd_threshold=0.75, max_conformers=None, conformer_max_attempts=1000, conformer_max_iter=200):
    '''
    Desciption: Prepares multiple molecules from SMILES strings, generates conformers, and returns lists of molecules, conformer IDs, and templates.
    Parameters:
    - smiles_list: list of str (SMILES strings for molecules)
    - conformer_rmsd_threshold: float (default: 0.75, RMSD threshold for conformer clustering)
    - max_conformers: int (default: None, maximum number of conformers to generate)
    - conformer_max_attempts: int (default: 1000, maximum attempts for conformer generation)
    - conformer_max_iter: int (default: 200, maximum iterations for conformer optimization)
    Output:
    tuple: (list of rdkit.Chem.rdchem.Mol (molecules with conformers), list of list of int (conformer IDs for each molecule), list of rdkit.Chem.rdchem.Mol (templates for each molecule))
    '''
    mol_list = []
    conf_ids_list = []
    templates_list = []
    
    for smiles in smiles_list:
        mol = read_smiles(smiles)
        mol_with_h, conf_ids = generate_diverse_conformers(
            mol, 
            rmsd_threshold=conformer_rmsd_threshold, 
            max_conformers=max_conformers, 
            max_attempts=conformer_max_attempts, 
            max_iter=conformer_max_iter
        )

        template = Chem.Mol(mol_with_h)  # 保存干净模板
        mol_list.append(mol_with_h)
        conf_ids_list.append(conf_ids)
        templates_list.append(template)
        print(f"分子 {len(mol_list)} 生成了 {len(conf_ids)} 个构象")
    
    return mol_list, conf_ids_list, templates_list


def merge_multiple_molecules(mol_templates, conf_ids, distance=30.0, random_transform=False):
    """合并多个分子（1~3个），1个分子时直接返回"""
    if len(mol_templates) == 1:
        # 单个分子无需合并
        mol = Chem.Mol(mol_templates[0])
        conf_id = conf_ids[0]
        return mol, conf_id
    
    # 多个分子时逐步合并（先合并前两个，再和第三个合并）
    merged_mol, merged_conf_id = None, None
    for i in range(len(mol_templates)):
        mol = Chem.Mol(mol_templates[i])
        conf_id = conf_ids[i]
        
        if i == 0:
            merged_mol = mol
            merged_conf_id = conf_id
            continue
        
        # 合并当前分子与已合并分子
        merged_mol, merged_conf_id = merge_two_molecules(
            merged_mol, merged_conf_id,
            mol, conf_id,
            distance=distance,
            random_transform=random_transform
        )
    
    Chem.SanitizeMol(merged_mol)
    return merged_mol, merged_conf_id


def merge_two_molecules(mol1, conf_id1, mol2, conf_id2, distance=30.0, random_transform=False):
    """合并两个分子（修复坐标保存问题）"""
    if random_transform:
        mol2 = rotate_molecule(
            mol2, conf_id2,
            angle_x=random.uniform(0, 360),
            angle_y=random.uniform(0, 360),
            angle_z=random.uniform(0, 360)
        )
        mol2 = translate_molecule(mol2, conf_id2, [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
    
    # 计算质心并调整距离
    centroid1 = calculate_centroid(mol1, conf_id1)
    centroid2 = calculate_centroid(mol2, conf_id2)
    direction = centroid1 - centroid2
    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else np.array([1.0, 0.0, 0.0])
    translation = direction * (distance - np.linalg.norm(centroid1 - centroid2))
    mol2 = translate_molecule(mol2, conf_id2, translation)
    
    # 创建合并分子容器
    merged_mol = Chem.RWMol()
    atom_map1 = {}  # 单独维护两个分子的原子映射，避免混淆
    atom_map2 = {}
    
    # 添加第一个分子的原子和键
    for atom in mol1.GetAtoms():
        new_atom = Atom(atom.GetSymbol())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
        idx = merged_mol.AddAtom(new_atom)
        atom_map1[atom.GetIdx()] = idx  # 原始原子索引 -> 合并后索引
    for bond in mol1.GetBonds():
        begin_idx = atom_map1[bond.GetBeginAtomIdx()]
        end_idx = atom_map1[bond.GetEndAtomIdx()]
        merged_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
    
    # 添加第二个分子的原子和键
    for atom in mol2.GetAtoms():
        new_atom = Atom(atom.GetSymbol())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
        idx = merged_mol.AddAtom(new_atom)
        atom_map2[atom.GetIdx()] = idx  # 原始原子索引 -> 合并后索引
    for bond in mol2.GetBonds():
        begin_idx = atom_map2[bond.GetBeginAtomIdx()]
        end_idx = atom_map2[bond.GetEndAtomIdx()]
        merged_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
    
    # 关键修复：显式创建与原子数匹配的构象并设置坐标
    num_atoms = merged_mol.GetNumAtoms()  # 获取合并后总原子数
    new_conformer = Chem.Conformer(num_atoms)  # 显式指定构象原子数
    merged_conf_id = merged_mol.AddConformer(new_conformer, assignId=True)
    merged_conf = merged_mol.GetConformer(merged_conf_id)  # 获取新构象
    
    # 从第一个分子复制坐标（严格对应原子映射）
    conf1 = mol1.GetConformer(conf_id1)
    for atom_idx in range(mol1.GetNumAtoms()):
        new_idx = atom_map1[atom_idx]  # 确保映射正确
        pos = conf1.GetAtomPosition(atom_idx)
        merged_conf.SetAtomPosition(new_idx, (pos.x, pos.y, pos.z))  # 显式设置坐标
    
    # 从第二个分子复制坐标（严格对应原子映射）
    conf2 = mol2.GetConformer(conf_id2)
    for atom_idx in range(mol2.GetNumAtoms()):
        new_idx = atom_map2[atom_idx]  # 确保映射正确
        pos = conf2.GetAtomPosition(atom_idx)
        merged_conf.SetAtomPosition(new_idx, (pos.x, pos.y, pos.z))  # 显式设置坐标
    
    # 确保分子结构正确
    Chem.SanitizeMol(merged_mol)
    return merged_mol, merged_conf_id


def prepare_constraints(
    merged_mol, 
    hbond_pairs, 
    equilibrium_ratio=1.6, 
    min_equilibrium_ratio=0.8,
    max_equilibrium_ratio=1.3,
    force_constant=5.0
):
    """根据氢键对准备约束条件"""
    single_bond_lengths = {
        ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43, ('C', 'H'): 1.09,
        ('N', 'C'): 1.47, ('N', 'N'): 1.45, ('N', 'O'): 1.41, ('N', 'H'): 1.01,
        ('O', 'C'): 1.43, ('O', 'N'): 1.41, ('O', 'O'): 1.48, ('O', 'H'): 0.96,
        ('H', 'C'): 1.09, ('H', 'N'): 1.01, ('H', 'O'): 0.96, ('H', 'H'): 0.74
    }
    equilibrium_distances = {k: v * equilibrium_ratio for k, v in single_bond_lengths.items()}
    
    constraints = []
    for (map1, map2) in hbond_pairs:
        idx1, idx2 = None, None
        for i in range(merged_mol.GetNumAtoms()):
            if merged_mol.GetAtomWithIdx(i).GetAtomMapNum() == map1:
                idx1 = i
            if merged_mol.GetAtomWithIdx(i).GetAtomMapNum() == map2:
                idx2 = i
            if idx1 is not None and idx2 is not None:
                break
        if idx1 is None or idx2 is None:
            raise ValueError(f"找不到原子映射号: {map1} 或 {map2}")
        
        elem1 = merged_mol.GetAtomWithIdx(idx1).GetSymbol()
        elem2 = merged_mol.GetAtomWithIdx(idx2).GetSymbol()
        eq_dist = equilibrium_distances.get((elem1, elem2), equilibrium_distances.get((elem2, elem1)))
        if eq_dist is None:
            raise ValueError(f"找不到 {elem1}-{elem2} 的平衡距离")
        
        constraints.append((idx1, idx2, False, eq_dist * min_equilibrium_ratio, eq_dist * max_equilibrium_ratio, force_constant))
    return constraints, equilibrium_distances


def check_and_collect_satisfied_structures(merged_mol, conf_id, hbond_pairs, equilibrium_distances, attempt_info, max_allowed_ratio=1.4):
    """检查约束是否满足并收集结构信息"""
    map_to_idx = {
        merged_mol.GetAtomWithIdx(i).GetAtomMapNum(): i
        for i in range(merged_mol.GetNumAtoms())
        if merged_mol.GetAtomWithIdx(i).GetAtomMapNum() != 0
    }
    
    all_satisfied = True
    results = []
    for (map1, map2) in hbond_pairs:
        idx1, idx2 = map_to_idx[map1], map_to_idx[map2]
        elem1, elem2 = merged_mol.GetAtomWithIdx(idx1).GetSymbol(), merged_mol.GetAtomWithIdx(idx2).GetSymbol()
        max_allowed = equilibrium_distances.get((elem1, elem2), equilibrium_distances.get((elem2, elem1))) * max_allowed_ratio
        
        pos1 = merged_mol.GetConformer(conf_id).GetAtomPosition(idx1)
        pos2 = merged_mol.GetConformer(conf_id).GetAtomPosition(idx2)
        distance = np.sqrt((pos1.x - pos2.x)** 2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)** 2)
        
        satisfied = distance < max_allowed
        all_satisfied = all_satisfied and satisfied
        results.append({
            "pair": (map1, map2), "elements": (elem1, elem2),
            "distance": distance, "max_allowed": max_allowed, "satisfied": satisfied
        })
        print(f"距离检查: {map1}({elem1})-{map2}({elem2}): {distance:.3f}Å (允许: {max_allowed:.3f}Å) - {'满足' if satisfied else '不满足'}")
    
    if not all_satisfied:
        return None
    
    # 收集满足条件的结构信息
    # elements = np.array([int_atom(atom.GetSymbol()) for atom in merged_mol.GetAtoms()], dtype=np.int32)
    elements = np.array([atom.GetSymbol() for atom in merged_mol.GetAtoms()], dtype=str)
    conf = merged_mol.GetConformer(conf_id)
    coordinates = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(merged_mol.GetNumAtoms())
    ], dtype=np.float64)
    
    return {
        "elements": elements,
        "coordinates": coordinates,
        "conf_id": conf_id,
        "rdmol": merged_mol,  # 新增：保存rdmol对象
        **attempt_info  # 包含构象ID、尝试次数等元信息
    }


def filter_diverse_structures(structures, rmsd_threshold=0.75, reorder=True):
    """基于RMSD筛选结构多样化的分子"""
    if not structures:
        return []

    print(f"共 {len(structures)} 个候选结构")
    print("选择第一个进行初始化...")
    selected = [structures[0]]
    
    for struct in structures[1:]:
        min_rmsd = min([
            calculate_rmsd_directly(
                struct["elements"], struct["coordinates"],
                s["elements"], s["coordinates"],
                # reorder=False # H 的位置会不对齐，后续力场优化对齐
                reorder=reorder # H 的位置会不对齐
            ) for s in selected
        ])
        if min_rmsd > rmsd_threshold:
            selected.append(struct)
            print(f"结构入选（与已选最小RMSD: {min_rmsd:.3f} > {rmsd_threshold}），当前数量: {len(selected)}")
    
    return selected


def save_structures(structures, output_dir, base_output_file, charge=0, multiplicity=1):
    """保存筛选后的结构到指定文件夹下的XYZ文件"""
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    for i, struct in enumerate(structures):
        # 构建完整的输出文件路径
        filename = f"{base_output_file}_{i+1}"
        output_file = os.path.join(output_dir, filename)
        
        write_xyz_file(
            struct["elements"],
            struct["coordinates"],
            struct["conf_id"],
            output_file,
            charge,
            multiplicity
        )
        print(f"已保存结构 {i+1} 到 {output_file}")



def process_reaction_pairs(diverse_structures, hbond_pairs, output_dir="output_pairs", 
                          charge=0, multiplicity=1):
    """
    处理新格式的键变化，直接设置键级并优化
    
    参数:
        diverse_structures: 筛选后的多样化结构列表
        hbond_pairs: 包含键变化信息的字典，格式为
                    {'new_bonds': [...], 'changed_bonds': [(map1, map2, bond_order), ...]}
        output_dir: 输出目录
    """
    # 键级到RDKit键类型的映射
    bond_order_map = {
        0.0: None,  # 断裂键
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC  # 芳香键特殊处理
    }

    # 从hbond_pairs中提取电荷变化并转换为字典 {原子映射号: 目标电荷}
    charge_changes = {}
    if 'changed_charges' in hbond_pairs:
        charge_changes = {map_num: target_charge for map_num, target_charge in hbond_pairs['changed_charges']}
    
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
       
    # 遍历每个多样化结构
    for pair_idx, struct in enumerate(diverse_structures):
        pair_dir = os.path.join(output_dir, f"rxn_pair{pair_idx+1}")
        os.makedirs(pair_dir, exist_ok=True)
        
        try:
            # 1. 保存反应前结构（原始结构）
            reactant_filename = os.path.join(pair_dir, "reactant")
            write_xyz_file(
                struct["elements"],
                struct["coordinates"],
                struct["conf_id"],
                reactant_filename,
                charge,
                multiplicity
            )
            print(f"已保存反应前结构到 {reactant_filename}.xyz")
            
            # 2. 从原始结构创建反应后结构（键编辑）
            if "rdmol" not in struct:
                raise ValueError("结构中未找到rdmol对象，请先在check_and_collect_satisfied_structures中保存rdmol")
            
            # 关键修复1：创建深拷贝避免原始分子污染
            product_mol = Chem.RWMol(Chem.Mol(struct["rdmol"]))
            atom_map_to_idx = {
                product_mol.GetAtomWithIdx(i).GetAtomMapNum(): i
                for i in range(product_mol.GetNumAtoms())
                if product_mol.GetAtomWithIdx(i).GetAtomMapNum() != 0
            }
            
            # 记录断裂的键（用于后续约束）
            broken_bonds = []
            # 应用键变化
            for (map1, map2, bond_order) in hbond_pairs['changed_bonds']:
                if map1 not in atom_map_to_idx or map2 not in atom_map_to_idx:
                    print(f"跳过无效键对: ({map1}, {map2})")
                    continue
                
                idx1 = atom_map_to_idx[map1]
                idx2 = atom_map_to_idx[map2]
                bond_type = bond_order_map.get(bond_order)
                
                # 移除现有键
                existing_bond = product_mol.GetBondBetweenAtoms(idx1, idx2)
                if existing_bond:
                    product_mol.RemoveBond(idx1, idx2)
                
                # 添加新键
                if bond_type is not None:
                    product_mol.AddBond(idx1, idx2, bond_type)
                    print(f"设置键 {map1}-{map2} 键级: {bond_order}")
                else:
                    print(f"断裂键 {map1}-{map2}")
                    broken_bonds.append((map1, map2))  # 记录断裂的键对

            # 2. 应用从hbond_pairs读取的电荷变化
            for map_num, target_charge in charge_changes.items():
                if map_num in atom_map_to_idx:
                    atom_idx = atom_map_to_idx[map_num]
                    product_mol.GetAtomWithIdx(atom_idx).SetFormalCharge(int(target_charge))  # 转为整数电荷
                    print(f"应用电荷变化: 原子 {map_num} → {target_charge}")

            product_mol = product_mol.GetMol()  # 从RWMol转换为Mol
            Chem.SanitizeMol(product_mol)  # 这一步会初始化环信息并检查结构有效性


            # 5. 为断裂键添加限制性优化约束
            print(f"为反应后结构 {pair_idx+1} 添加断裂键约束...")
            # 准备断裂键的约束条件（复用主函数的约束逻辑）
            constraints, _ = prepare_constraints(
                product_mol, 
                broken_bonds,  # 对断裂的键添加约束
                equilibrium_ratio=1.6,  # 断裂键的平衡距离比例更大（原键长的2倍）
                min_equilibrium_ratio=0.8,
                max_equilibrium_ratio=1.3,
                force_constant=5.0  # 如果逆反应这里得很大如 50.0
            )


            # 4. 直接优化（不添加约束）
            # print(f"优化反应后结构 {pair_idx+1}...")
            # 添加氢原子（确保优化正确性）
            # product_mol = Chem.AddHs(product_mol)
            # 创建新构象（复制原始坐标作为初始值）
            conf_id = product_mol.AddConformer(struct["rdmol"].GetConformer(), assignId=True)
            # 直接优化（不使用约束）
            # AllChem.MMFFOptimizeMolecule(product_mol, confId=conf_id, maxIters=500)

            # 执行限制性优化（复用主函数的优化逻辑）
            print(f"对反应后结构 {pair_idx+1} 进行限制性优化...")
            optimized_mol, optimized_conf_id = optimize_with_constraints(
                product_mol, 
                conf_id, 
                constraints,
                max_iter=300
            )
            
            # 6. 收集并保存产物结构
            elements = np.array([atom.GetSymbol() for atom in optimized_mol.GetAtoms()], dtype=str)
            conf = optimized_mol.GetConformer(optimized_conf_id)
            coordinates = np.array([
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(optimized_mol.GetNumAtoms())
            ], dtype=np.float64)
            
            product_filename = os.path.join(pair_dir, "product")
            write_xyz_file(
                elements,
                coordinates,
                optimized_conf_id,
                product_filename,
                charge,
                multiplicity
            )
            print(f"已保存反应后结构到 {product_filename}.xyz")
            
        except Exception as e:
            print(f"处理反应对 {pair_idx+1} 时出错: {str(e)}")
            continue




# ------------------------------
# 主函数（仅包含流程控制）
# ------------------------------
def gen_conformers_and_opt_complex(
    smiles_list, 
    rxn_pairs, 
    conformer_rmsd_threshold=-1.0,
    max_conformers=None, 
    conformer_max_attempts=1000,
    conformer_max_iter=200,
    output_dir="output_structures", 
    base_output_file="optimized_complex", 
    complex_max_attempts=10, 
    complex_initial_distance=20.0,
    charge=0, 
    multiplicity=1, 
    rmsd_threshold=0.75,
    equilibrium_ratio=1.6, 
    min_equilibrium_ratio=0.8,
    max_equilibrium_ratio=1.3,
    force_constant=5.0,
    max_allowed_ratio=1.4,
    filter_rmsd_threshold=0.75,
    reorder=True
):
    """
    主函数：处理1~3个SMILES分子，生成优化后的复合物结构
    
    参数:
        smiles_list: SMILES字符串列表（长度1~3）
        rxn_pairs: 氢键作用对列表，如[(map1, map2), ...]
        base_output_file: 输出文件前缀
        max_attempts: 每个构象组合的最大尝试次数
        charge: 分子总电荷
        multiplicity: 自旋多重度
        rmsd_threshold: 结构多样性筛选的RMSD阈值
    Description: processes 1-3 SMILES strings, generates conformers, merges them into complexes, applies constraints based on reaction pairs, optimizes structures, filters for diversity, and saves results.
    Parameters:
        smiles_list: list of str, SMILES list(length 1-3)
        rxn_pairs: list of tuple, rxn pairs,[(map1, map2), ...]
        base_output_file: str, base output file name
        max_attempts: int, max attempts
        charge: int, charge
        multiplicity: int, multiplicity
        rmsd_threshold: float, rmsd threshold
    Returns: None
    """
    # 1. 输入验证
    if not (1 <= len(smiles_list) <= 3):
        raise ValueError("smiles_list长度必须为1~3")
    
    # 2. 准备分子和构象
    print("===== 准备分子和构象 =====")
    mol_list, conf_ids_list, templates_list = prepare_molecules_and_conformers(smiles_list, conformer_rmsd_threshold, max_conformers, conformer_max_attempts, conformer_max_iter)
    
    # 3. 遍历构象组合，尝试生成满足条件的结构
    print("\n===== 生成并优化复合物 =====")
    satisfied_structures = []
    total_attempts = 0
    max_total_attempts = complex_max_attempts * np.prod([len(cids) for cids in conf_ids_list])
    
    # 生成所有构象组合（笛卡尔积）
    from itertools import product
    conf_combinations = product(*conf_ids_list)
    
    for conf_ids in conf_combinations:
        attempt = 0
        while attempt < complex_max_attempts and total_attempts < max_total_attempts:
            attempt += 1
            total_attempts += 1
            print(f"\n----- 构象组合 {conf_ids} - 尝试 {attempt}/{complex_max_attempts}（总尝试: {total_attempts}）-----")
            
            try:
                # a. 合并分子（根据数量自动处理）
                merged_mol, merged_conf_id = merge_multiple_molecules(
                    templates_list,  # 使用干净模板避免累积变换
                    conf_ids,
                    distance=complex_initial_distance,
                    random_transform=(attempt > 1)  # 第一次不随机，后续随机
                )
                
                # b. 准备约束条件
                constraints, equilibrium_distances = prepare_constraints(
                    merged_mol, 
                    rxn_pairs['new_bonds'],
                    equilibrium_ratio=equilibrium_ratio, 
                    min_equilibrium_ratio=min_equilibrium_ratio,
                    max_equilibrium_ratio=max_equilibrium_ratio,
                    force_constant=force_constant
                )
                
                # c. 限制性优化
                print("进行限制性优化...")
                optimized_mol, optimized_conf_id = optimize_with_constraints(
                    merged_mol, merged_conf_id, constraints
                )
                
                # d. 检查约束并收集结构
                attempt_info = {
                    "conf_ids": conf_ids,
                    "attempt": attempt,
                    "total_attempt": total_attempts
                }
                valid_struct = check_and_collect_satisfied_structures(
                    optimized_mol, optimized_conf_id, rxn_pairs['new_bonds'], equilibrium_distances, attempt_info,
                    max_allowed_ratio=max_allowed_ratio
                )
                
                if valid_struct:
                    satisfied_structures.append(valid_struct)
                    print("当前组合满足条件，已记录")
                
            except Exception as e:
                print(f"尝试失败: {str(e)}，继续重试...")
                continue
    
    if not satisfied_structures:
        print("\n未找到满足条件的结构！")
        return
    
    # 4. 筛选结构多样性
    print("\n===== 筛选结构多样性 =====")
    diverse_structures = filter_diverse_structures(
        satisfied_structures, 
        rmsd_threshold=filter_rmsd_threshold, 
        reorder=reorder
    )
    print(f"筛选完成，保留 {len(diverse_structures)} 个多样化结构")

    ######
    # TODO:
    # check_and_collect_satisfied_structures 可以保存下 rdmol 对象
    # 此处添加遍历保留下的多样化结构，筛选可以严格一些，因为 NEB 几乎肯定能成功
    # 然后用这些结构进行 rdmol editmol 成断键以后，简单优化
    # 并保存 pairs 信息以备 UMA 及后续 CINEB 使用
    # 保存成 forward/rxn_pair1/reactant.xyz forward/rxn_pair1/product.xyz 这样的结构
    ######

    print("\n===== 生成反应前后结构对 =====")
    process_reaction_pairs(
        diverse_structures,
        rxn_pairs,  # 新格式的键变化字典
        output_dir=os.path.join(output_dir, "reaction_pairs")
    )

    # # 5. 保存结果
    # print("\n===== 保存结果 =====")
    # save_structures(diverse_structures, output_dir, base_output_file, charge, multiplicity)
    # print(f"\n全部完成！共保存 {len(diverse_structures)} 个结构")
