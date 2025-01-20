# Pytcoil

Pytcoil consists of some utilities that could stream out massive amount of tcoil layouts that are ready for ASITIC and EMX simulations, and even for ML applications. This can be quite useful for preparing training dataset for some neural networks such as the one in https://github.com/ChrisZonghaoLi/upcnn.

- "asitic": it contains the scripts that are used to prepare ASITIC t-coil EM simulation results for training neural network. Most scripts are based on a mock 7nm FinFET process.
- "common": it consists of some scripts that will be used by both "asitic" and "emx", such as equivalent circuit extration. 
- "emx": it consists of the scripts that are used to prepare EMX t-coil EM simulation results for training neural network. It also has some SKILL script generator which can be used to massively stream out lots of t-coil layout from Cadence Virtuoso. Most of the scripts are based on GF22-FDSOI.
