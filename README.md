--------------------------------------------------------------------------------
<span style="font-size:larger;">packaging-extrapolation Manual</span>
========
![Static Badge](https://img.shields.io/badge/Extrapolation_method-blue)
![Static Badge](https://img.shields.io/badge/Quantum_Chemistry-red)
![Static Badge](https://img.shields.io/badge/Basis_set-green)
![doi](https://img.shields.io/badge/Chemical_energy-yellow)

## About

* This package contains partial extrapolation methods in quantum chemistry, written using the extrapolation method proposed in the literature. Extrapolation to the CBS limit can be done by entering two successive energies.

## Quickly Use

* Please use the `pip` command to install: `pip install packaging_extrapolation` or `python3 -m pip install packaging_extrapolation`
* Please make sure the package is the latest: `pip install --upgrade packaging_extrapolation`
* After installation, test the example in `src/packaging_extrapolation/examples/examples_energy.py` to see if you get results.
  * Extrapolation Method Calls: `python  examples_energy.py -m "Klopper_1986" -xe -76.0411795 -ye -76.0603284 -low 2 -high 3 -a 4.25`
  * `-m`: extrapolation method name.
  * `-xe`: energy for E(X).
  * `-ye`: energy for E(Y).
  * `-low`: cardinal number for X.
  * `-high`: cardinal number for Y.
  * `-a`: extrapolation parameter alpha/beta.

## Ten Extrapolation Schemes

| Method                      | Two-point From                                               | Name            |                          Reference                           |
| :-------------------------- | :----------------------------------------------------------- | --------------- | :----------------------------------------------------------: |
| Klopper-1986                | $E_{CBS}=\frac{E(Y)e^{-α\sqrt{X}}-E(X)e^{-α\sqrt{Y}}}{e^{-α\sqrt{X}}-e^{-α\sqrt{Y}}}$ | `Klopper_1986`  | [https://doi.org/10.1016/0166-1280(86)80068-9](https://doi.org/10.1016/0166-1280(86)80068-9) |
| Feller-1992                 | $E_{CBS}=\frac{E(Y)e^{-αX}-E(X)e^{-αY}}{e^{-αX}-e^{-αY}}$    | `Feller_1992`   | [https://doi.org/10.1063/1.462652](https://doi.org/10.1063/1.462652) |
| Truhlar-1998 (Hartree-Fock) | $E_{CBS}=\frac{E(Y)X^{-\alpha}-E(X)Y^{-\alpha}}{X^{-\alpha}-Y^{-\alpha}}$ | `Truhlar_1998`  | [https://doi.org/10.1016/S0009-2614(98)00866-5](https://doi.org/10.1016/S0009-2614(98)00866-5) |
| Jensen-2001                 | $E_{CBS}=\frac{E(Y)(X+1)e^{-α\sqrt{X}}-E(X)(Y+1))e^{-α\sqrt{Y}}}{(X+1)e^{-α\sqrt{X}}-(Y+1)e^{-α\sqrt{Y}}}$ | `Jensen_2001`   |              https://doi.org/10.1063/1.1413524               |
| Schwenke-2005               | $E_{CBS}=[E(Y)-E(X)]\alpha+E(X)$                             | `Schwenke_2005` |              https://doi.org/10.1063/1.1824880               |
| Martin-1996                 | $E_{CBS}=\frac{E(Y)(X+1/2)^{-\beta}-E(X)(Y+1/2)^{-\beta}}{(X+1/2)^{-\beta}-(Y+1/2)^{-\beta}}$ | `Martin_1996`   |         https://doi.org/10.1016/0009-2614(96)00898-6         |
| Truhlar-1998 (Correlation)  | $E_{CBS}=\frac{E(Y)X^{-\beta}-E(X)Y^{-\beta}}{X^{-\beta}-Y^{-\beta}}$ | `Truhlar_1998`  | [https://doi.org/10.1016/S0009-2614(98)00866-5](https://doi.org/10.1016/S0009-2614(98)00866-5) |
| Huh-2003                    | $E_{CBS}=\frac{E(Y)(X+\beta)^{-3}-E(X)(Y+\beta)^{-3}}{(X+\beta)^{-3}-(Y+\beta)^{-3}}$ | `HuhLee_2003`   |              https://doi.org/10.1063/1.1534091               |
| Bakowies-2007               | $E_{CBS}=\frac{E(Y)(X+1)^{-\beta}-E(X)(Y+1)^{-\beta}}{(X+1)^{-\beta}-(Y+1)^{-\beta}}$ | `Bkw_2007`      |              https://doi.org/10.1063/1.2749516               |
| OAN(C)                      | $E_{CBS}=\frac{3^3E(Y)-\beta^3E(X)}{3^3-\beta^3}$            | `OAN_C`         |              https://doi.org/10.1002/jcc.23896               |



## Another Use

* If you need to calculate the extrapolation energy of more systems, please refer to the following examples:

```python
from packaging_extrapolation import UtilTools
from packaging_extrapolation.Extrapolation import FitMethod
import pandas as pd
import numpy as np

"""
Calculate more systems.
"""

if __name__ == "__main__":
    # Input file.
    data = pd.read_csv('../data/hf.CSV')

    # Extrapolation model.
    model = FitMethod()
    # The E(X) and E(Y).
    x_energy_list, y_energy_list = data['aug-cc-pvdz'], data['aug-cc-pvtz']
    # Using Klopper-1986 method and alpha=4.25, extrapolate to the CBS limit at the AV {D, T}Z basis set pair.
    low_card, high_card, alpha, method_name = 2, 3, 4.25, 'Klopper_1986'
    result = UtilTools.train_alpha(model=model,
                                   method=method_name,
                                   x_energy_list=x_energy_list,
                                   y_energy_list=y_energy_list,
                                   low_card=low_card,
                                   high_card=high_card,
                                   alpha=alpha)
    for i in range(len(result)):
        print(result[i], 'Eh')
    df = pd.DataFrame()
    df['CBS Energy'] = result
    # Output file.
    df.to_csv('CBS_Energy.csv', index=False)
```

* The input file should be in `.csv` format and have the following content:

```python
mol,aug-cc-pvdz,aug-cc-pvtz
HCN,-92.8880397,-92.9100033
HCO,-113.2672513,-113.2947633
HNO,-129.8114596,-129.8401888
HO2,-150.2024221,-150.239531
N2O,-183.7105405,-183.7530387
NH2,-55.5749363,-55.5878344
NH3,-56.1972947,-56.2127423
NO2,-204.0664514,-204.1137363
```



| Functions                                                    |
| ------------------------------------------------------------ |
| `UtilTools.calc_MAD(y_true, y_pred)`: Calculate the Mean Absolute Deviation (kcal/mol). |
| `UtilTools.calc_max_MAD(y_true, y_pred)`: Calculate the Maximum Absolute Deviation (kcal/mol). |
| `UtilTools.calc_min_MAD(y_true, y_pred)`: Calculate the Minimum Absolute Deviation (kcal/mol). |
| `UtilTools.calc_RMSE(y_true, y_pred)`: Calculate the Root Mean Square Deviation (kcal/mol). |
| `UtilTools.calc_MSD(y_true, y_pred)`: Calculate the Mean Square Deviation (kcal/mol). |
| `UtilTools.calc_MaxPosMAD(y_true, y_pred)`: Calculate the Maximum Positive Deviation (kcal/mol). |
| `UtilTools.train_alpha(*,  model, method, x_energy_list, y_energy_list, alpha, low_card, high_card)`: Calculate extrapolated energy. |
| `UtilLog.extract_energy(input_path, output_path)`: Extracting energy from many log files. |
| `train_all(*, model, method, x_energy_list, y_energy_list, low_card, high_card, limit_list, init_guess=0.001, temp='RMSD')` : Optimizing extrapolation parameters with `RMSD` or `MAD`. |



