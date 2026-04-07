import os
import shutil
import glob
import json
import sys
sys.path.append("./tools")
from typing import Dict, Any

import random
import numpy as np

from MapReaction import map_reaction
from ParseReactionMapIndex import parse_reaction_map_index
from GenConformersAndOptComplex import gen_conformers_and_opt_complex
from PreOptComplex import batch_optimize_xyz
# from RunCINEB import run_cineb
from RunCINEB_ash import run_cineb


def load_config_from_json(json_path: str) -> Dict[str, Any]:
    """从JSON文件加载所有配置参数"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证必要字段
        required_fields = [
            "reactants", "products", "seed", "direction", 
            "base_results_dir", "main_force_field", "target_force_field",
            "charge", "multiplicity", "num_images",
            # 新增的必要参数
            "conformer_rmsd_threshold", "max_conformers", 
            "conformer_max_attempts", "conformer_max_iter",
            "complex_max_attempts", "complex_initial_distance",
            "rmsd_threshold", "equilibrium_ratio", "min_equilibrium_ratio",
            "max_equilibrium_ratio", "force_constant", "max_allowed_ratio",
            "filter_rmsd_threshold", "reorder", "opt_fmax", "opt_steps", "task_name",
            "climb", "neb_fmax", "neb_steps"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件缺少必要字段: {field}")
        
        # 验证字段类型
        if not isinstance(config["reactants"], list) or not isinstance(config["products"], list):
            raise ValueError("'reactants' 和 'products' 必须是列表")
        if not isinstance(config["seed"], int):
            raise ValueError("'seed' 必须是整数")
        
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")
    except Exception as e:
        raise ValueError(f"加载配置失败: {str(e)}")


def main():
    """主函数：从配置文件加载参数，执行工作流"""
    # 命令行指定配置文件路径（可根据需要修改默认路径）
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config.json"
    
    # 加载配置
    config = load_config_from_json(config_path)
    print("成功加载配置参数")

    # 设置随机数种子
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    print(f"已设置随机数种子: {seed}")

    # 解析核心参数
    reactants = config["reactants"]
    products = config["products"]
    direction = config["direction"]
    base_results_dir = config["base_results_dir"]
    main_force_field = config["main_force_field"]
    target_force_field = config["target_force_field"]
    charge = config["charge"]
    mult = config["multiplicity"]
    num_images = config["num_images"]

    # 解析gen_conformers_and_opt_complex所需的新参数
    conformer_rmsd_threshold = config["conformer_rmsd_threshold"]
    max_conformers = config["max_conformers"]
    conformer_max_attempts = config["conformer_max_attempts"]
    conformer_max_iter = config["conformer_max_iter"]
    complex_max_attempts = config["complex_max_attempts"]
    complex_initial_distance = config["complex_initial_distance"]
    rmsd_threshold = config["rmsd_threshold"]
    equilibrium_ratio = config["equilibrium_ratio"]
    min_equilibrium_ratio = config["min_equilibrium_ratio"]
    max_equilibrium_ratio = config["max_equilibrium_ratio"]
    force_constant = config["force_constant"]
    max_allowed_ratio = config["max_allowed_ratio"]
    filter_rmsd_threshold = config["filter_rmsd_threshold"]
    reorder = config["reorder"]

    # 解析 opt 所需参数
    opt_fmax = config["opt_fmax"]
    opt_steps = config["opt_steps"]
    task_name = config["task_name"]

    # 解析 neb 所需参数
    climb = config["climb"]
    neb_fmax = config["neb_fmax"]
    neb_steps = config["neb_steps"]

    # 打印加载信息
    print(f"Reactants: {reactants}")
    print(f"Products: {products}")
    print(f"反应方向: {direction}")
    print(f"主力场: {main_force_field}, 目标力场: {target_force_field}")

    # 执行反应映射
    results = map_reaction(reactants, products)
    print("成功完成反应的原子映射")

    # 解析反应位点
    mapped_rxn = results[0]["mapped_rxn"]
    rxn_site_data = parse_reaction_map_index(mapped_rxn)
    print("成功完成反应位点解析")

    # 处理SMILES列表和反应对
    mapped_rxn_update = rxn_site_data[0]['updated_smiles']
    smiles_list = mapped_rxn_update.split(">>")[0].split(".") if direction == 'forward' else mapped_rxn_update.split(">>")[1].split(".")
    rxn_pairs = rxn_site_data[0].get(direction, [])

    # 生成输出目录
    output_dir = os.path.join(base_results_dir, direction, main_force_field)
    base_output_file = "reactant_complex" if direction == 'forward' else "product_complex"

    # 打印路径信息
    print(f"SMILES列表包含 {len(smiles_list)} 个分子")
    print(f"{direction} 方向包含 {len(rxn_pairs)} 个反应对")
    print(f"输出根目录: {output_dir}")

    # 生成构象并优化（主力场），传入所有新参数
    gen_conformers_and_opt_complex(
        smiles_list=smiles_list,
        rxn_pairs=rxn_pairs,
        conformer_rmsd_threshold=conformer_rmsd_threshold,
        max_conformers=max_conformers,
        conformer_max_attempts=conformer_max_attempts,
        conformer_max_iter=conformer_max_iter,
        output_dir=output_dir,
        base_output_file=base_output_file,
        complex_max_attempts=complex_max_attempts,
        complex_initial_distance=complex_initial_distance,
        charge=charge,
        multiplicity=mult,
        rmsd_threshold=rmsd_threshold,
        equilibrium_ratio=equilibrium_ratio,
        min_equilibrium_ratio=min_equilibrium_ratio,
        max_equilibrium_ratio=max_equilibrium_ratio,
        force_constant=force_constant,
        max_allowed_ratio=max_allowed_ratio,
        filter_rmsd_threshold=filter_rmsd_threshold,
        reorder=reorder
    )

    # 处理目标力场（UMA）的优化和CINEB计算
    reaction_pairs_dir = os.path.join(output_dir, "reaction_pairs")
    rxn_pair_dirs = glob.glob(os.path.join(reaction_pairs_dir, "rxn_pair*"))

    for dir_path in rxn_pair_dirs:
        if not os.path.isdir(dir_path):
            continue  # 确保是目录
        
        # 生成目标力场路径（替换主力场为目标力场）
        target_dir_path = dir_path.replace(main_force_field, target_force_field)
        os.makedirs(target_dir_path, exist_ok=True)  # 确保目录存在
        print(f"\n处理反应对目录: {dir_path}")
        print(f"目标力场目录: {target_dir_path}")

        # 批量优化（目标力场）
        batch_optimize_xyz(
            input_dir=dir_path,
            output_dir=target_dir_path,
            charge=charge,
            mult=mult,
            fmax=opt_fmax,
            steps=opt_steps,
            task_name=task_name
        )

        # 执行CINEB计算
        reactant_file = os.path.join(target_dir_path, "reactant.xyz")
        product_file = os.path.join(target_dir_path, "product.xyz")
        
        if os.path.exists(reactant_file) and os.path.exists(product_file):
            ts_guess = run_cineb(
                reactant_file=reactant_file,
                product_file=product_file,
                output_file=target_dir_path,
                num_images=num_images,
                charge=charge,
                mult=mult,
                climb=climb, 
                fmax=neb_fmax, 
                steps=neb_steps, 
                task_name=task_name
            )
            print(f"已完成 {os.path.basename(dir_path)} 的CINEB计算")

            ## 使用 ash 有一堆中间文件不知道怎么处理，简单移动一下 ##
            # 文件清理和移动操作
            current_dir = os.getcwd()  # 获取当前执行目录（run.py所在目录）
            
            # 1. 删除当前目录下所有 image*/ 文件夹
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and item.startswith("image"):
                    try:
                        shutil.rmtree(item_path)
                        print(f"已删除目录: {item_path}")
                    except Exception as e:
                        print(f"删除目录 {item_path} 失败: {e}")
            
            # 2. 移动指定类型文件到 target_dir_path
            os.makedirs(target_dir_path, exist_ok=True)
            
            file_patterns = ['*.result', '*.xyz', '*.interp', '*.energy']
            for pattern in file_patterns:
                # 关键修复：使用 glob.glob() 而不是 glob()
                for file_path in glob.glob(os.path.join(current_dir, pattern)):
                    target_file = os.path.join(target_dir_path, os.path.basename(file_path))
                    try:
                        shutil.move(file_path, target_file)
                        print(f"已移动文件: {file_path} -> {target_file}")
                    except Exception as e:
                        print(f"移动文件 {file_path} 失败: {e}")
                ####

        else:
            print(f"警告: {target_dir_path} 中缺少反应物或产物文件，跳过CINEB计算")

    print("\n所有任务完成")


if __name__ == "__main__":
    main()
# import os
# import shutil
# import glob
# import json
# from typing import Dict, Any

# import random
# import numpy as np

# from MapReaction import map_reaction
# from ParseReactionMapIndex import parse_reaction_map_index
# from GenConformersAndOptComplex import gen_conformers_and_opt_complex
# from PreOptComplex import batch_optimize_xyz
# # from RunCINEB import run_cineb
# from RunCINEB_ash import run_cineb


# def load_config_from_json(json_path: str) -> Dict[str, Any]:
#     """从JSON文件加载所有配置参数"""
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             config = json.load(f)
        
#         # 验证必要字段
#         required_fields = [
#             "reactants", "products", "seed", "direction", 
#             "base_results_dir", "main_force_field", "target_force_field",
#             "charge", "multiplicity", "num_images",
#             # 新增的必要参数
#             "conformer_rmsd_threshold", "max_conformers", 
#             "conformer_max_attempts", "conformer_max_iter",
#             "complex_max_attempts", "complex_initial_distance",
#             "rmsd_threshold", "equilibrium_ratio", "min_equilibrium_ratio",
#             "max_equilibrium_ratio", "force_constant", "max_allowed_ratio",
#             "filter_rmsd_threshold", "reorder", "opt_fmax", "opt_steps", "task_name",
#             "climb", "neb_fmax", "neb_steps"
#         ]
#         for field in required_fields:
#             if field not in config:
#                 raise ValueError(f"配置文件缺少必要字段: {field}")
        
#         # 验证字段类型
#         if not isinstance(config["reactants"], list) or not isinstance(config["products"], list):
#             raise ValueError("'reactants' 和 'products' 必须是列表")
#         if not isinstance(config["seed"], int):
#             raise ValueError("'seed' 必须是整数")
        
#         return config
        
#     except json.JSONDecodeError as e:
#         raise ValueError(f"JSON解析错误: {str(e)}")
#     except Exception as e:
#         raise ValueError(f"加载配置失败: {str(e)}")


# def main(config_path: str = "./config.json"):
#     """主函数：从配置文件加载参数，执行工作流"""
#     # 加载配置
#     config = load_config_from_json(config_path)
#     print(f"成功加载配置参数: {config_path}")

#     # 设置随机数种子
#     seed = config["seed"]
#     random.seed(seed)
#     np.random.seed(seed)
#     print(f"已设置随机数种子: {seed}")

#     # 解析核心参数
#     reactants = config["reactants"]
#     products = config["products"]
#     direction = config["direction"]
#     base_results_dir = config["base_results_dir"]
#     main_force_field = config["main_force_field"]
#     target_force_field = config["target_force_field"]
#     charge = config["charge"]
#     mult = config["multiplicity"]
#     num_images = config["num_images"]

#     # 解析gen_conformers_and_opt_complex所需的新参数
#     conformer_rmsd_threshold = config["conformer_rmsd_threshold"]
#     max_conformers = config["max_conformers"]
#     conformer_max_attempts = config["conformer_max_attempts"]
#     conformer_max_iter = config["conformer_max_iter"]
#     complex_max_attempts = config["complex_max_attempts"]
#     complex_initial_distance = config["complex_initial_distance"]
#     rmsd_threshold = config["rmsd_threshold"]
#     equilibrium_ratio = config["equilibrium_ratio"]
#     min_equilibrium_ratio = config["min_equilibrium_ratio"]
#     max_equilibrium_ratio = config["max_equilibrium_ratio"]
#     force_constant = config["force_constant"]
#     max_allowed_ratio = config["max_allowed_ratio"]
#     filter_rmsd_threshold = config["filter_rmsd_threshold"]
#     reorder = config["reorder"]

#     # 解析 opt 所需参数
#     opt_fmax = config["opt_fmax"]
#     opt_steps = config["opt_steps"]
#     task_name = config["task_name"]

#     # 解析 neb 所需参数
#     climb = config["climb"]
#     neb_fmax = config["neb_fmax"]
#     neb_steps = config["neb_steps"]

#     # 打印加载信息
#     print(f"Reactants: {reactants}")
#     print(f"Products: {products}")
#     print(f"反应方向: {direction}")
#     print(f"主力场: {main_force_field}, 目标力场: {target_force_field}")

#     # 执行反应映射
#     results = map_reaction(reactants, products)
#     print("成功完成反应的原子映射")

#     # 解析反应位点
#     mapped_rxn = results[0]["mapped_rxn"]
#     rxn_site_data = parse_reaction_map_index(mapped_rxn)
#     print("成功完成反应位点解析")

#     # 处理SMILES列表和反应对
#     mapped_rxn_update = rxn_site_data[0]['updated_smiles']
#     smiles_list = mapped_rxn_update.split(">>")[0].split(".") if direction == 'forward' else mapped_rxn_update.split(">>")[1].split(".")
#     rxn_pairs = rxn_site_data[0].get(direction, [])

#     # 生成输出目录
#     output_dir = os.path.join(base_results_dir, direction, main_force_field)
#     base_output_file = "reactant_complex" if direction == 'forward' else "product_complex"

#     # 打印路径信息
#     print(f"SMILES列表包含 {len(smiles_list)} 个分子")
#     print(f"{direction} 方向包含 {len(rxn_pairs)} 个反应对")
#     print(f"输出根目录: {output_dir}")

#     # 生成构象并优化（主力场）
#     gen_conformers_and_opt_complex(
#         smiles_list=smiles_list,
#         rxn_pairs=rxn_pairs,
#         conformer_rmsd_threshold=conformer_rmsd_threshold,
#         max_conformers=max_conformers,
#         conformer_max_attempts=conformer_max_attempts,
#         conformer_max_iter=conformer_max_iter,
#         output_dir=output_dir,
#         base_output_file=base_output_file,
#         complex_max_attempts=complex_max_attempts,
#         complex_initial_distance=complex_initial_distance,
#         charge=charge,
#         multiplicity=mult,
#         rmsd_threshold=rmsd_threshold,
#         equilibrium_ratio=equilibrium_ratio,
#         min_equilibrium_ratio=min_equilibrium_ratio,
#         max_equilibrium_ratio=max_equilibrium_ratio,
#         force_constant=force_constant,
#         max_allowed_ratio=max_allowed_ratio,
#         filter_rmsd_threshold=filter_rmsd_threshold,
#         reorder=reorder
#     )

#     # 处理目标力场（UMA）的优化和CINEB计算
#     reaction_pairs_dir = os.path.join(output_dir, "reaction_pairs")
#     rxn_pair_dirs = glob.glob(os.path.join(reaction_pairs_dir, "rxn_pair*"))

#     for dir_path in rxn_pair_dirs:
#         if not os.path.isdir(dir_path):
#             continue  # 确保是目录
        
#         # 生成目标力场路径（替换主力场为目标力场）
#         target_dir_path = dir_path.replace(main_force_field, target_force_field)
#         os.makedirs(target_dir_path, exist_ok=True)  # 确保目录存在
#         print(f"\n处理反应对目录: {dir_path}")
#         print(f"目标力场目录: {target_dir_path}")

#         # 批量优化（目标力场）
#         batch_optimize_xyz(
#             input_dir=dir_path,
#             output_dir=target_dir_path,
#             charge=charge,
#             mult=mult,
#             fmax=opt_fmax,
#             steps=opt_steps,
#             task_name=task_name
#         )

#         # 执行CINEB计算
#         reactant_file = os.path.join(target_dir_path, "reactant.xyz")
#         product_file = os.path.join(target_dir_path, "product.xyz")
        
#         if os.path.exists(reactant_file) and os.path.exists(product_file):
#             ts_guess = run_cineb(
#                 reactant_file=reactant_file,
#                 product_file=product_file,
#                 output_file=target_dir_path,
#                 num_images=num_images,
#                 charge=charge,
#                 mult=mult,
#                 climb=climb, 
#                 fmax=neb_fmax, 
#                 steps=neb_steps, 
#                 task_name=task_name
#             )
#             print(f"已完成 {os.path.basename(dir_path)} 的CINEB计算")

#             ## 使用 ash 有一堆中间文件不知道怎么处理，简单移动一下 ##
#             current_dir = os.getcwd()  # 获取当前执行目录（run.py所在目录）
            
#             # 1. 删除当前目录下所有 image*/ 文件夹
#             for item in os.listdir(current_dir):
#                 item_path = os.path.join(current_dir, item)
#                 if os.path.isdir(item_path) and item.startswith("image"):
#                     try:
#                         shutil.rmtree(item_path)
#                         print(f"已删除目录: {item_path}")
#                     except Exception as e:
#                         print(f"删除目录 {item_path} 失败: {e}")
            
#             # 2. 移动指定类型文件到 target_dir_path
#             os.makedirs(target_dir_path, exist_ok=True)
            
#             file_patterns = ['*.result', '*.xyz', '*.interp', '*.energy']
#             for pattern in file_patterns:
#                 for file_path in glob.glob(os.path.join(current_dir, pattern)):
#                     target_file = os.path.join(target_dir_path, os.path.basename(file_path))
#                     try:
#                         shutil.move(file_path, target_file)
#                         print(f"已移动文件: {file_path} -> {target_file}")
#                     except Exception as e:
#                         print(f"移动文件 {file_path} 失败: {e}")

#         else:
#             print(f"警告: {target_dir_path} 中缺少反应物或产物文件，跳过CINEB计算")

#     print("\n所有任务完成")


# if __name__ == "__main__":
#     # 在这里直接指定配置文件路径
#     main(config_path="./config.json")
