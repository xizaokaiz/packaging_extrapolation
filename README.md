--------------------------------------------------------------------------------
<span style="font-size:larger;">packaging_extrapolation Manual</span>
========
![Static Badge](https://img.shields.io/badge/Extrapolation_method-blue)
![Static Badge](https://img.shields.io/badge/Quantum_Chemistry-red)
![Static Badge](https://img.shields.io/badge/Basis_set-green)
![Static Badge](https://img.shields.io/badge/Chemical_energy-yellow)

# About
* This package contains partial extrapolation methods in quantum chemistry.
* This package is written in the extrapolation method proposed in the literature, using the Python language. Extrapolation to the CBS limit can be done by entering two successive energies.

# Installation
* Please use the 'pip' command to install: `pip install packaging_extrapolation` or `python3 -m pip install packaging_extrapolation`
* Please make sure the package is update: `pip install --upgrade packaging_extrapolation`
* After installation, test the example in `src/packaging_extrapolation/examples/examples_energy.py` to see if you get results.
  * Extrapolation Method Calls：`python  examples_energy.py -m "Klopper_1986" -xe -76.0411795 -ye -76.0603284 -low 2 -high 3 -a 4.25`
  * `-m`: extrapolation method name.
  * `-xe`: energy for E(X).
  * `-ye`: energy for E(Y).
  * `-low`: cardinal number for X.
  * `-high`: cardinal number for Y.
  * `-a`: extrapolation parameter alpha.

# 关于
* 集成了量子化学中的部分外推法
* 由于量子化学的计算成本与基组相关，其随着基组序列逐渐递增。
本工具库以文献提出的外推公式所编写，使用Python语言。
通过输入两个水平的能量即可外推到CBS极限值。

# 安装方法
* 使用`pip`命令安装：`pip install packaging_extrapolation` 或者 `python3 -m pip install packaging_extrapolation`
* 请保证库的最新状态：`pip install --upgrade packaging_extrapolation`
* 安装完成后，请测试`src/packaging_extrapolation/examples_energy.py` 的实例脚本，是否可以得到结果。
  * 调用外推模型：`python  examples_energy.py -m "Klopper_1986" -xe -76.0411795 -ye -76.0603284 -low 2 -high 3 -a 4.25`
  * -m: extrapolation method name.
  * -xe: energy for E(X).
  * -ye: energy for E(Y).
  * -low: cardinal number for X.
  * -high: cardinal number for Y.
  * -a: extrapolation parameter alpha.