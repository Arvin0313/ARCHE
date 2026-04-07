from ase import Atoms
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import os

def run_cineb(reactant_file, product_file, output_file, num_images=7, charge=0, mult=1, climb=True, fmax=0.05, steps=300, task_name='omol'):
    """
    使用CINEB方法寻找过渡态
    
    参数:
        reactant_file: 反应物XYZ文件路径
        product_file: 生成物XYZ文件路径
        num_images: 插值图像数量（包括反应物和产物）
        charge: 体系电荷
        mult: 体系自旋多重度
    """

    # 跳过已存在的输出文件（可选）
    if os.path.exists(f'{output_file}/uma_ts_guess.xyz'):
        print(f"已存在，跳过: {output_file}")
        return None
            
    # 读取反应物和产物结构
    reactant = read(reactant_file)
    product = read(product_file)
    reactant.info.update({"spin": mult, "charge": charge})
    product.info.update({"spin": mult, "charge": charge})
    
    # 确保原子数量和种类一致
    assert len(reactant) == len(product), "反应物和产物的原子数量必须相同"
    assert sorted(reactant.get_chemical_symbols()) == sorted(product.get_chemical_symbols()), \
        "反应物和产物的原子种类必须相同"
    
    # 创建图像列表
    images = [reactant]
    # 添加中间图像
    for i in range(num_images - 2):
        image = reactant.copy()
        images.append(image)
    images.append(product)
    
    # 为每个图像创建独立的XTB计算器
    for image in images:
        # 每个图像都有自己的计算器实例
        calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda"), 
            task_name=task_name
        )
        image.calc = calc
    
    # 创建NEB对象（CINEB）
    # "aseneb" "improvedtangent" "eb" "spline" "string"
    # neb = NEB(images, climb=climb, parallel=False)  # climb=True启用CI-NEB
    
    # # 插值初始路径
    # neb.interpolate(method='idpp', mic=True)
    
    # # 创建优化器
    # optimizer = BFGS(neb) # (default value is 0.2 Å)
    # # optimizer = BFGS(neb, trajectory='aimnet2_neb.traj', logfile='aimnet2_neb.log')

    # # 运行优化
    # print("开始CINEB优化...")
    # # print("第一阶段")
    # # optimizer.k = 0.5
    # conv = optimizer.run(fmax=fmax, steps=steps)  # 力的收敛标准

    # if not conv:
    #     print("CINEB 收敛失败")
        # return None

    # print("第二阶段")
    # optimizer.k = 0.01
    # optimizer.maxstep = 0.001
    # conv = optimizer.run(fmax=0.06, steps=200)

    # if not conv:
    #     print("CINEB 收敛失败")
        # return None
    
    # 输出结果
    # nebtools = NEBTools(images)
    # Ef, dE = nebtools.get_barrier(fit=False, raw=False)
    # print(Ef, dE)

    # fig = nebtools.plot_bands()
    # fig.savefig('test.png')

#####

    neb_initial = NEB(images, climb=False, parallel=False)
    neb_initial.interpolate(method='idpp', mic=True)
    optimizer_initial = BFGS(neb_initial)
    print("开始初始NEB优化...")

    conv_initial = optimizer_initial.run(fmax=0.07, steps=200)

    neb_cineb = NEB(images, climb=True, parallel=False)
    optimizer_cineb = BFGS(neb_cineb, maxstep=0.02)
    print("开始CINEB优化...")
    conv_cineb = optimizer_cineb.run(fmax=0.05, steps=500)

#####
    # 计算所有图像的能量，找到能量最高的结构作为过渡态
    energies = []
    for i, image in enumerate(images):
        energy = image.get_potential_energy()
        energies.append(energy)
        # 处理numpy数组的情况
        if hasattr(energy, '__iter__'):
            energy_value = float(energy)
        else:
            energy_value = energy
        print(f"图像 {i} 的能量: {energy_value:.6f} eV")
    
    # 找到能量最高的图像索引
    ts_idx = energies.index(max(energies))
    print(f"\n能量最高的点为图像 {ts_idx}，作为过渡态初猜")
    
    # 保存结果
    write(f'{output_file}/uma_ts_guess.xyz', images[ts_idx])  # 保存能量最高的结构
    print(f"过渡态搜索完成，结果保存在ts_guess.xyz")
    # print(f"轨迹文件保存在aimnet2_neb.traj，日志保存在aimnet2_neb.log")
    
    return images[ts_idx]  # 返回过渡态猜测结构

