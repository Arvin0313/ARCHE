from ash import Fragment, FairchemTheory, NEB
# from ash import *
import os
import numpy as np

def run_cineb(reactant_file, product_file, output_file, num_images=8, charge=0, mult=1, 
                  climb=True, fmax=0.05, steps=300, task_name='omol'):
    """
    使用ash的CINEB方法寻找过渡态（基于ash的稳健算法实现）
    
    参数:
        reactant_file: 反应物XYZ文件路径
        product_file: 生成物XYZ文件路径
        num_images: 插值图像数量（包括反应物和产物）
        charge: 体系电荷
        mult: 体系自旋多重度
    """
    # 创建输出目录
    os.makedirs(output_file, exist_ok=True)
    
    # 跳过已存在的输出文件
    if os.path.exists(f'{output_file}/uma_ts_guess.xyz'):
        print(f"已存在，跳过: {output_file}")
        return None
    
    # 1. 读取反应物和产物结构（转换为ash的Molecule对象）
    reactant = Fragment(xyzfile=reactant_file, charge=charge, mult=mult)
    product = Fragment(xyzfile=product_file, charge=charge, mult=mult)
    
    # 2. 设置计算器（适配FAIRChem的MLIP模型）
    theory = FairchemTheory(model_name="uma-s-1p1", task_name=task_name, device="cuda")
    
    # 3.  配置CINEB计算（使用ash的NEB模块）
    NEB_result = NEB(
        reactant=reactant,
        product=product,
        theory=theory,
        images=num_images,
        CI=climb,
        maxiter=steps,
    )

    # print(NEB_result)