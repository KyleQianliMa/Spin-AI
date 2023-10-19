# Spin-AI

Spin-AI is a prototype being actively developed to model inelastic neutron scattering data. It is construct by interfering an auto encoder with a feed-forward neural network.

# Motivation - Yb2O3

The $Yb_2O_3$ is a binary system with complex chiral structural. The $Yb$ is the only magnetic ion in the system yet they occupy two different symmetry sites(See picture below). The spin Hamiltonian is governed by two exchange interactions $J_1$ and $J_2$. Symmetry allows 12 free parameters to poppulate two 3 $\times$ 3 matrices for the exchange interactions. Fitting these parameters with inelastic neutron scattering data has been challenging.

![alt text](pictures/Picture1.png)

For example, how would you fit a spin wave data that looks like this:

  <img src="https://github.com/KyleQianliMa/Spin-AI/tree/main/pictures/Picture3.png" width="10">
