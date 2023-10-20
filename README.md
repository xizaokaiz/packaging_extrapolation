# 关于`packaging_extrapolation`
* 集成了量子化学中的部分外推法
* 由于量子化学的计算成本与基组相关，其随着基组序列逐渐递增。
本工具库以文献提出的外推公式所编写，使用Python语言。
通过输入两个水平的能量即可外推到CBS极限值。

## 安装方法
* 使用`pip`命令安装：`pip install packaging_extrapolation` 或者 `python3 -m pip install packaging_extrapolation`
* 请保持库的最新状态：`pip install --upgrade packaging_extrapolation`
* 安装完成后，请测试`src/packaging_extrapolation/examples` 中的实例脚本，是否可以得到结果。