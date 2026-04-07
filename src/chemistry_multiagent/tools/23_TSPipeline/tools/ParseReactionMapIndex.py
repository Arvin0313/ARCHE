from rdkit import Chem
from rdkit.Chem import rdchem
from typing import List, Set, Tuple, Dict
import math

def get_atom_map_number(atom: rdchem.Atom) -> int:
    """获取原子的映射编号"""
    try:
        return int(atom.GetProp("molAtomMapNumber"))
    except:
        return -1  # 无映射编号的原子返回-1

def set_atom_map_number(atom: rdchem.Atom, map_num: int) -> None:
    """为原子设置映射编号"""
    atom.SetProp("molAtomMapNumber", str(map_num))

def get_bonds_with_maps(mol: rdchem.Mol) -> Set[Tuple[int, int, float]]:
    """从分子中提取所有带有映射编号的原子间的键（包含键级）"""
    bonds = set()
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        map1 = get_atom_map_number(atom1)
        map2 = get_atom_map_number(atom2)
        
        if map1 != -1 and map2 != -1 and map1 != map2:
            sorted_maps = tuple(sorted([map1, map2]))
            # 转换键类型为键级
            bond_type = bond.GetBondType()
            if bond_type == rdchem.BondType.SINGLE:
                bond_order = 1.0
            elif bond_type == rdchem.BondType.DOUBLE:
                bond_order = 2.0
            elif bond_type == rdchem.BondType.TRIPLE:
                bond_order = 3.0
            elif bond_type == rdchem.BondType.AROMATIC:
                bond_order = 1.5  # 芳香键按1.5处理
            else:
                bond_order = 0.0  # 其他类型暂不考虑
            bonds.add((sorted_maps[0], sorted_maps[1], bond_order))
    return bonds

def count_attached_hydrogens(atom: rdchem.Atom) -> int:
    """计算原子所连接的氢原子数量（包括显式和隐式）"""
    return atom.GetTotalNumHs()

def get_hydrogen_counts(mol_list: List[rdchem.Mol]) -> Dict[int, int]:
    """获取每个重原子（非H）连接的氢原子数量"""
    h_counts = {}
    for mol in mol_list:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                map_num = get_atom_map_number(atom)
                if map_num != -1:
                    h_counts[map_num] = count_attached_hydrogens(atom)
    return h_counts

def find_hydrogen_transfers(reactants: List[rdchem.Mol], products: List[rdchem.Mol]) -> List[Tuple[int, int]]:
    """检测氢原子转移：从重原子A转移到重原子B"""
    reactant_h = get_hydrogen_counts(reactants)
    product_h = get_hydrogen_counts(products)
    
    all_heavy_atoms = set(reactant_h.keys()).union(set(product_h.keys()))
    h_changes = {atom: product_h.get(atom, 0) - reactant_h.get(atom, 0) for atom in all_heavy_atoms}
    
    h_donors = []  # 失去H的原子
    h_acceptors = []  # 获得H的原子
    
    for atom, change in h_changes.items():
        if change < 0:
            h_donors.extend([atom] * abs(change))
        elif change > 0:
            h_acceptors.extend([atom] * change)
    
    return list(zip(h_donors, h_acceptors))

def get_next_available_map_number(mols: List[rdchem.Mol]) -> int:
    """获取下一个可用的原子映射号"""
    max_map = 0
    for mol in mols:
        for atom in mol.GetAtoms():
            map_num = get_atom_map_number(atom)
            if map_num > max_map:
                max_map = map_num
    return max_map + 1

def add_hydrogen_maps(reactants: List[rdchem.Mol], products: List[rdchem.Mol], h_transfers: List[Tuple[int, int]]) -> Tuple[List[rdchem.Mol], List[rdchem.Mol]]:
    """为转移的氢原子添加映射号，返回修改后的反应物和产物列表"""
    next_map = get_next_available_map_number(reactants + products)
    
    # 处理反应物：为供体原子添加带映射的显式H
    new_reactants = []
    for mol in reactants:
        emol = Chem.EditableMol(mol)
        to_add = []
        donor_info = {}

        for atom in mol.GetAtoms():
            atom_map = get_atom_map_number(atom)
            if atom_map == -1:
                continue
            is_donor = any(donor == atom_map for donor, _ in h_transfers)
            if is_donor:
                donor_count = sum(1 for d, _ in h_transfers if d == atom_map)
                donor_info[atom_map] = {
                    "original_idx": atom.GetIdx(),
                    "total_h": atom.GetNumExplicitHs() + atom.GetNumImplicitHs(),
                    "donate_h": donor_count
                }

                for _ in range(donor_count):
                    h_atom = Chem.Atom("H")
                    set_atom_map_number(h_atom, next_map)
                    to_add.append((atom.GetIdx(), h_atom, next_map))
                    next_map += 1
        
        for atom_idx, h_atom, h_map in to_add:
            h_idx = emol.AddAtom(h_atom)
            emol.AddBond(atom_idx, h_idx, rdchem.BondType.SINGLE)
        
        new_mol = emol.GetMol()
        for atom in new_mol.GetAtoms():
            atom_map = get_atom_map_number(atom)
            if atom_map in donor_info:
                info = donor_info[atom_map]
                new_explicit_h = info["total_h"] - info["donate_h"]
                atom.SetNumExplicitHs(new_explicit_h)
                atom.SetProp("_NumImplicitHs", str(0))
                atom.UpdatePropertyCache()
        
        try:
            Chem.SanitizeMol(new_mol)
        except:
            for atom in new_mol.GetAtoms():
                if atom.GetExplicitValence() > atom.GetMaxValence():
                    atom.SetNumExplicitHs(atom.GetMaxValence() - atom.GetTotalValence() + atom.GetNumExplicitHs())
            Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        
        new_reactants.append(new_mol)
    
    # 处理产物：为受体原子添加带映射的显式H
    new_products = []
    for mol in products:
        emol = Chem.EditableMol(mol)
        to_add = []
        acceptor_info = {}

        for atom in mol.GetAtoms():
            atom_map = get_atom_map_number(atom)
            if atom_map == -1:
                continue
                
            is_acceptor = any(acceptor == atom_map for _, acceptor in h_transfers)
            if is_acceptor:
                acceptor_count = sum(1 for _, a in h_transfers if a == atom_map)
                acceptor_info[atom_map] = {
                    "original_idx": atom.GetIdx(),
                    "total_h": atom.GetNumExplicitHs() + atom.GetNumImplicitHs(),
                    "receive_h": acceptor_count
                }
                
                for i in range(acceptor_count):
                    h_atom = Chem.Atom("H")
                    h_map = None
                    for idx, (d, a) in enumerate(h_transfers):
                        if a == atom_map and idx == i:
                            h_map = get_next_available_map_number(reactants) + idx
                            break
                    if h_map:
                        set_atom_map_number(h_atom, h_map)
                        to_add.append((atom.GetIdx(), h_atom))
        
        for atom_idx, h_atom in to_add:
            h_idx = emol.AddAtom(h_atom)
            emol.AddBond(atom_idx, h_idx, rdchem.BondType.SINGLE)
        
        new_mol = emol.GetMol()
        for atom in new_mol.GetAtoms():
            atom_map = get_atom_map_number(atom)
            if atom_map in acceptor_info:
                info = acceptor_info[atom_map]
                new_explicit_h = info["total_h"] - info["receive_h"]
                atom.SetNumExplicitHs(new_explicit_h)
                atom.SetProp("_NumImplicitHs", str(0))
                atom.UpdatePropertyCache()
        
        try:
            Chem.SanitizeMol(new_mol)
        except:
            for atom in new_mol.GetAtoms():
                if atom.GetExplicitValence() > atom.GetMaxValence():
                    atom.SetNumExplicitHs(atom.GetMaxValence() - atom.GetTotalValence() + atom.GetNumExplicitHs())
            Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        
        new_products.append(new_mol)
    
    return new_reactants, new_products

def get_atom_charges(mol_list: List[rdchem.Mol]) -> Dict[int, float]:
    """获取每个原子的形式电荷"""
    charges = {}
    for mol in mol_list:
        for atom in mol.GetAtoms():
            map_num = get_atom_map_number(atom)
            if map_num != -1:
                # 获取形式电荷并转换为float类型
                charges[map_num] = float(atom.GetFormalCharge())
    return charges

def parse_reaction_smiles(reaction_smiles: str) -> Tuple[List[rdchem.Mol], List[rdchem.Mol]]:
    """解析反应SMILES，返回反应物和产物的分子列表"""
    if ">>" not in reaction_smiles:
        raise ValueError("无效的反应SMILES，必须包含'>>'分隔符")
    
    reactants_smiles, products_smiles = reaction_smiles.split(">>")
    reactants = [Chem.MolFromSmiles(smi, sanitize=False) for smi in reactants_smiles.split(".")]
    products = [Chem.MolFromSmiles(smi, sanitize=False) for smi in products_smiles.split(".")]
    
    reactants = [mol for mol in reactants if mol is not None]
    products = [mol for mol in products if mol is not None]
    
    for mol in reactants + products:
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
    
    return reactants, products

def analyze_reaction(reaction_smiles: str) -> Tuple[
    List[Tuple[int, int]], 
    List[Tuple[int, int]],
    List[Tuple[int, int, float]],
    List[Tuple[int, int, float]],
    List[Tuple[int, float]],
    List[Tuple[int, float]],
    str
]:
    """分析反应中的键变化和电荷变化"""
    reactants, products = parse_reaction_smiles(reaction_smiles)
    h_transfers = find_hydrogen_transfers(reactants, products)
    reactants_with_h, products_with_h = add_hydrogen_maps(reactants, products, h_transfers)
    
    # 收集包含键级的键信息
    reactant_bonds = set()
    for mol in reactants_with_h:
        reactant_bonds.update(get_bonds_with_maps(mol))
    
    product_bonds = set()
    for mol in products_with_h:
        product_bonds.update(get_bonds_with_maps(mol))
    
    # 提取原子对（不带键级）
    reactant_pairs = {(a, b) for (a, b, _) in reactant_bonds}
    product_pairs = {(a, b) for (a, b, _) in product_bonds}
    
    # 计算新键和断键
    new_bonds = [(a, b) for (a, b) in product_pairs if (a, b) not in reactant_pairs]  # 产物新增的键
    broken_bonds = [(a, b) for (a, b) in reactant_pairs if (a, b) not in product_pairs]  # 反应物断裂的键
    new_bonds.sort()
    broken_bonds.sort()
    
    # 构建键级映射字典
    reactant_bond_map = {(a, b): order for (a, b, order) in reactant_bonds}
    product_bond_map = {(a, b): order for (a, b, order) in product_bonds}
    
    # 键级变化的共同键（反应前后都存在但键级不同）
    common_pairs = reactant_pairs & product_pairs
    changed_common_pairs = [
        (a, b) for (a, b) in common_pairs 
        if not math.isclose(reactant_bond_map[(a, b)], product_bond_map[(a, b)])
    ]
    
    # 正向反应的changed_bonds：新键（产物键级） + 键级变化的共同键（产物键级） + 断键（0.0）
    forward_changed = [
        (a, b, product_bond_map[(a, b)]) for (a, b) in new_bonds
    ] + [
        (a, b, product_bond_map[(a, b)]) for (a, b) in changed_common_pairs
    ] + [
        (a, b, 0.0) for (a, b) in broken_bonds  # 断键用0.0表示
    ]
    forward_changed.sort()
    
    # 逆向反应的changed_bonds：断键（反应物键级） + 键级变化的共同键（反应物键级） + 新键（0.0）
    reverse_changed = [
        (a, b, reactant_bond_map[(a, b)]) for (a, b) in broken_bonds
    ] + [
        (a, b, reactant_bond_map[(a, b)]) for (a, b) in changed_common_pairs
    ] + [
        (a, b, 0.0) for (a, b) in new_bonds  # 原新键在逆向中为断键，用0.0表示
    ]
    reverse_changed.sort()
    
    # 分析电荷变化
    reactant_charges = get_atom_charges(reactants_with_h)
    product_charges = get_atom_charges(products_with_h)
    
    # 找出所有有映射的原子
    all_atoms = set(reactant_charges.keys()).union(set(product_charges.keys()))
    
    # 检测正向反应的电荷变化（产物中的电荷值）
    forward_changed_charges = []
    # 检测逆向反应的电荷变化（反应物中的电荷值）
    reverse_changed_charges = []
    
    for atom in all_atoms:
        reactant_charge = reactant_charges.get(atom, 0.0)
        product_charge = product_charges.get(atom, 0.0)
        
        if not math.isclose(reactant_charge, product_charge):
            forward_changed_charges.append((atom, product_charge))
            reverse_changed_charges.append((atom, reactant_charge))
    
    forward_changed_charges.sort()
    reverse_changed_charges.sort()
    
    # 生成更新后的反应SMILES
    reactants_smiles = ".".join([Chem.MolToSmiles(mol) for mol in reactants_with_h])
    products_smiles = ".".join([Chem.MolToSmiles(mol) for mol in products_with_h])
    updated_reaction_smiles = f"{reactants_smiles}>>{products_smiles}"
    
    return (new_bonds, broken_bonds, forward_changed, reverse_changed,
            forward_changed_charges, reverse_changed_charges, updated_reaction_smiles)


def parse_reaction_map_index(mapped_reaction):
    (new_bonds, broken_bonds, forward_changed, reverse_changed,
     forward_changed_charges, reverse_changed_charges, updated_smiles) = analyze_reaction(mapped_reaction)

    result = [
        {
            "forward": {
                "new_bonds": new_bonds,
                "changed_bonds": forward_changed,
                "changed_charges": forward_changed_charges
            },
            "reverse": {
                "new_bonds": broken_bonds,
                "changed_bonds": reverse_changed,
                "changed_charges": reverse_changed_charges
            },
            "updated_smiles": updated_smiles
        }
    ]
    
    return result


def analyze_reaction_natural(reaction_smiles: str):
    """
    分析反应SMILES中的键变化和电荷变化，并生成自然语言化学机理描述。
    返回：
        new_bonds, broken_bonds, forward_changed, reverse_changed,
        forward_changed_charges, reverse_changed_charges, updated_reaction_smiles,
        summary_text
    """
    from collections import defaultdict

    # 调用原来的分析函数获取数据
    (new_bonds, broken_bonds, forward_changed, reverse_changed,
     forward_changed_charges, reverse_changed_charges, updated_reaction_smiles) = analyze_reaction(reaction_smiles)

    # 为每个原子创建标识：元素符号 + 分子序号 + 原子索引
    reactants, products = parse_reaction_smiles(reaction_smiles)
    all_mols = reactants + products
    atom_labels = {}
    for mol_idx, mol in enumerate(all_mols, start=1):
        for atom in mol.GetAtoms():
            map_num = get_atom_map_number(atom)
            if map_num != -1:
                atom_labels[map_num] = f"{atom.GetSymbol()}{map_num}"

    # 构建文字描述
    lines = [f"反应分析: {reaction_smiles}\n"]

    # 新键
    if new_bonds:
        lines.append("在反应中形成的新键:")
        for a, b in new_bonds:
            lines.append(f"  {atom_labels.get(a,a)} 与 {atom_labels.get(b,b)} 形成了新键")
    else:
        lines.append("反应中没有新键形成。")

    # 断裂键
    if broken_bonds:
        lines.append("\n断裂的键:")
        for a, b in broken_bonds:
            lines.append(f"  {atom_labels.get(a,a)} 与 {atom_labels.get(b,b)} 的键断裂")
    else:
        lines.append("\n没有键断裂。")

    # 键级变化
    changed_bonds = [ (a,b,order) for (a,b,order) in forward_changed if order > 0.0 and (a,b) not in new_bonds ]
    if changed_bonds:
        lines.append("\n键级变化的键:")
        for a, b, order in changed_bonds:
            lines.append(f"  {atom_labels.get(a,a)} - {atom_labels.get(b,b)} 的键级变化为 {order}")

    # 电荷变化
    if forward_changed_charges:
        lines.append("\n电荷变化:")
        for atom, charge in forward_changed_charges:
            lines.append(f"  {atom_labels.get(atom,atom)} 的电荷变为 {charge:+.1f}")

    summary_text = "\n".join(lines)
    return (new_bonds, broken_bonds, forward_changed, reverse_changed,
            forward_changed_charges, reverse_changed_charges, updated_reaction_smiles,
            summary_text)
