--------------------------------------------------------------------------------
<span style="font-size:larger;">packaging-extrapolation Manual</span>
========
![Static Badge](https://img.shields.io/badge/Extrapolation_method-blue)
![Static Badge](https://img.shields.io/badge/Quantum_Chemistry-red)
![Static Badge](https://img.shields.io/badge/Basis_set-green)
![doi](https://img.shields.io/badge/Chemical_energy-yellow)

## About

* This package contains partial extrapolation methods in quantum chemistry.
* This package is written using the extrapolation method proposed in the literature. Extrapolation to the CBS limit can be done by entering two successive energies.

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
  * `-a`: extrapolation parameter alpha.

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







