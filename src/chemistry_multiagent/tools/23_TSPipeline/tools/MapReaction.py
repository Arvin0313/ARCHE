from rxnmapper import RXNMapper
from typing import List, Dict, Any

def map_reaction(reactants: List[str], products: List[str]) -> List[Dict[str, Any]]:
    """
    使用RXNMapper对一个反应的反应物和产物进行原子映射
    
    参数:
        reactants: 反应物SMILES列表，每个元素是一个或多个反应物SMILES
        products: 产物SMILES列表，每个元素是一个或多个产物SMILES
        
    返回:
        包含映射结果的列表，每个元素是一个字典，包含:
        - mapped_rxn: 带原子映射的反应SMILES
        - confidence: 映射的置信度
        
    示例:
        输入:
            reactants = [
                "CC(C)S.CN(C)C=O",
                "Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]"
            ]
            products = [
                "CC(C)Sc1ncccc1F"
            ]
        
        输出:
            [
                {
                    "mapped_rxn": "[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]",
                    "confidence": 0.9565621227897111
                },
            ]
    """    
    # 构建反应SMILES
    rxn_smiles = [".".join(reactants) + ">>" + ".".join(products)]
    
    # 初始化RXNMapper并获取结果
    rxn_mapper = RXNMapper()
    results = rxn_mapper.get_attention_guided_atom_maps(rxn_smiles)
    
    # 格式化结果
    return [
        {
            "mapped_rxn": result.get("mapped_rxn", ""),
            "confidence": result.get("confidence", 0.0)
        }
        for result in results
    ]
