import os
import sys
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

def optimize_structure(input_xyz, output_xyz, charge=0, mult=1, fmax=0.01, steps=500, task_name="omol"):
    """
    使用FAIRChem计算器优化分子结构
    
    参数:
        input_xyz: 输入.xyz文件路径
        output_xyz: 优化后结构的输出路径
        charge: 分子整体电荷（默认0）
        mult: 自旋多重度（默认1）
    """
    try:
        # 读取初始结构
        atoms = read(input_xyz)
        atoms.info.update({"spin": mult, "charge": charge})

        # 初始化FAIRChem计算器
        calculator = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda"), 
            task_name=task_name
        )
        
        # 为原子结构关联计算器
        atoms.calc = calculator
        
        # 使用BFGS优化器进行结构优化
        optimizer = BFGS(atoms)
        convergence = optimizer.run(fmax=fmax, steps=steps)  # 收敛标准：最大力小于0.01 eV/Å
        
        if convergence:
            # 保存优化后的结构
            atoms.write(output_xyz)
            return True, f"优化成功: {output_xyz}"
        else:
            return False, f"优化失败：达到最大步数({steps})仍未收敛 - {input_xyz}"

    except Exception as e:
        return False, f"优化失败 {input_xyz}: {str(e)}"

def batch_optimize_xyz(input_dir, output_dir, charge=0, mult=1, fmax=0.01, steps=500, task_name="omol"):
    """
    批量优化文件夹中的所有XYZ文件
    
    参数:
        input_dir: 包含输入XYZ文件的文件夹路径
        output_dir: 保存优化后XYZ文件的目标文件夹路径
        charge: 分子整体电荷（默认0）
        mult: 自旋多重度（默认1）
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入文件夹不存在 - {input_dir}")
        return
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # print(filename)
        # 只处理XYZ文件
        if filename.lower().endswith(".xyz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 跳过已存在的输出文件（可选）
            if os.path.exists(output_path):
                print(f"已存在，跳过: {filename}")
                continue
            
            print(f"处理文件 {input_path}")
            # 调用核心优化函数
            success, message = optimize_structure(
                input_xyz=input_path,
                output_xyz=output_path,
                charge=charge,
                mult=mult,
                fmax=fmax,
                steps=steps,
                task_name=task_name
            )
            
            print(message)