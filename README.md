# ChebyKAN-as-a-Robust-Alternative-to-Traditional-KANs
ChebyKAN is an evolution of Kolmogorov–Arnold Networks, employing Chebyshev polynomials to approximate complex functions with improved stability and efficiency. It combines mathematical elegance with practical relevance, paving the way for scientific modelling, and robust financial systems.

<img width="894" height="545" alt="image" src="https://github.com/user-attachments/assets/39c3c80e-d3ae-4ff0-abbc-aa93af1f40ad" />

The training loss plot (MSE over 500 epochs on a logarithmic scale) visually demonstrates that ChebyKAN converges faster and achieves a significantly lower final error compared to the LinearSplineKAN and PolyKAN models. While the linear spline basis shows reasonable performance, the plain monomial basis (PolyKAN) struggles with ill-conditioning, resulting in slower convergence and a higher error plateau. This graphical comparison highlights the superior numerical stability and efficiency of Chebyshev polynomials as basis functions in KAN architectures, as ChebyKAN requires fewer epochs to reach a better solution.
