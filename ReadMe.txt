## Kernel combinations
`nlk.py` file implements a non linear combination of kernels based on the following paper: https://cs.nyu.edu/~mohri/pub/nlk.pdf

`simpleMKL.py` file implements the simpleMKL algorithm from this paper: http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf

## Notes
In practice both algorithms are not very useful. Indeed, in practise we seek to find the best combination of kernels that leads to the best performance after cross-validation, but both algorithms give the best combination of kernels possible during the training phase... because they seek the best combinations possible during the training phase...

## Advice
If you ever want to achieve the best possible results on your cross-validation sets, just use a polynomial __random combination__ of your kernels (the result is guaranteed to be a kernel). This is how my colleague https://github.com/rlespinet and I ranked 2/80 on the Kaggle for the Ecole Normale Supérieure.

_PS_: Go see all the kernels he implemented in C++ for the challenge here: https://github.com/rlespinet/rkernel/tree/master/lib. This guy is insane!