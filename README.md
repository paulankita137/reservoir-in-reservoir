Reservoirs-in-Reservoir

The repository implements the Reservoirs-in-Reservoir learning architecture of Recurrently connected Spiking Neural Networks. The learning described in the paper is applied to a set of aperiodic functions. The func.py file has the equation of the function. The train and test files are for training the networks and testing the networks respectively. The number of neurons, number of trials can be altered through the parameters in the respective train and test files to experiment with. The rate based and time-to-first-spike based implementations are available in separate folders. For any questions send an email to - ankita.paul@drexel.edu

The proposed training procedure consists of generating targets for both the recurrently-connected hidden layer and the output layer (i.e., for a full RSNN system), and using the recursive least square-based First-Order and Reduced Control Error (FORCE) algorithm to fit the activity of each layer to its target. We demonstrate the improved performance and noise robustness of the proposed full-FORCE training procedure to model 10 dynamic systems using RSNNs with leaky integrate and fire (LIF) neurons and spike rate-based encoding. For energy- efficient hardware implementation, an alternative time-to-first-spike (TTFS) encoding is implemented for the full-FORCE train-ing procedure. Compared to rate-based encoding, full-FORCE with TTFS encoding requires lower spike count and facilitate faster convergence to the target response.

In this repository we show the application of full-FORCE training for RSNNs to follow an accordian function. The rate based approach achieves lower error rates but higher spike rate :


The TTFS based approach achieves higher error rate compared to rate based but lower spike rate :

