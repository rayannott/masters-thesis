# some notes here

- when the RESOLUTION is increased, the CNN struggles to learn with the same kernel size (probably because the receptive field becomes too small then)
- why is the $u_{\text{init}}$ not periodic? (it should be, right?) *probably resolved*
- the Lyapunov exponent of an unoptimized model is negative -- the errors don't grow exponentially since the output of the model stays nearly always zero