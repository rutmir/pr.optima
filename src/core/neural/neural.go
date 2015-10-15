package neural

import "pr.optima/src/core/neural/mlpbase"

type MultiLayerPerceptron struct {
	innerobj *mlpbase.Multilayerperceptron
}

func NewMLP() *MultiLayerPerceptron {
	return &MultiLayerPerceptron{
		innerobj: mlpbase.NewMlp()}
}

func CreateMLP(mlp *mlpbase.Multilayerperceptron) *MultiLayerPerceptron {
	return &MultiLayerPerceptron{
		innerobj: mlp}
}

/*************************************************************************
Creates  neural  network  with  NIn  inputs,  NOut outputs, without hidden
layers, with linear output layer. Network weights are  filled  with  small
random values.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate0(nin, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreate0(nin, nout, network.innerobj)
	return network
}


/*************************************************************************
Same  as  MLPCreate0,  but  with  one  hidden  layer  (NHid  neurons) with
non-linear activation function. Output layer is linear.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate1(nin, nhid, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreate1(nin, nhid, nout, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreate0, but with two hidden layers (NHid1 and  NHid2  neurons)
with non-linear activation function. Output layer is linear.
 $ALL

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate2(nin, nhid1, nhid2, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreate2(nin, nhid1, nhid2, nout, network.innerobj)
	return network
}

/*************************************************************************
Creates  neural  network  with  NIn  inputs,  NOut outputs, without hidden
layers with non-linear output layer. Network weights are filled with small
random values.

Activation function of the output layer takes values:

	(B, +INF), if D>=0
or
	(-INF, B), if D<0.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateB0(nin, nout int, b, d float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreateb0(nin, nout, b, d, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateB0 but with non-linear hidden layer.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateB1(nin, nhid, nout int, b, d float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreateb1(nin, nhid, nout, b, d, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateB0 but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateB2(nin, nhid1, nhid2, nout int, b, d float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreateb2(nin, nhid1, nhid2, nout, b, d, network.innerobj)
	return network
}

/*************************************************************************
Creates  neural  network  with  NIn  inputs,  NOut outputs, without hidden
layers with non-linear output layer. Network weights are filled with small
random values. Activation function of the output layer takes values [A,B].

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateR0(nin, nout int, a, b float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreater0(nin, nout, a, b, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateR0, but with non-linear hidden layer.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateR1(nin, nhid, nout int, a, b float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreater1(nin, nhid, nout, a, b, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateR0, but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateR2(nin, nhid1, nhid2, nout int, a, b float64) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreater2(nin, nhid1, nhid2, nout, a, b, network.innerobj)
	return network
}

/*************************************************************************
Creates classifier network with NIn  inputs  and  NOut  possible  classes.
Network contains no hidden layers and linear output  layer  with  SOFTMAX-
normalization  (so  outputs  sums  up  to  1.0  and  converge to posterior
probabilities).

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreateC0(nin, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreatec0(nin, nout, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateC0, but with one non-linear hidden layer.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreateC1(nin, nhid, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreatec1(nin, nhid, nout, network.innerobj)
	return network
}

/*************************************************************************
Same as MLPCreateC0, but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreateC2(nin, nhid1, nhid2, nout int) *MultiLayerPerceptron {
	network := NewMLP()
	mlpbase.MlpCreatec2(nin, nhid1, nhid2, nout, network.innerobj)
	return network
}

/*************************************************************************
Randomization of neural network weights

  -- ALGLIB --
	 Copyright 06.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpRandomize(network *MultiLayerPerceptron) {
	mlpbase.MlpRandomize(network.innerobj)
}

/*************************************************************************
Randomization of neural network weights and standartisator

  -- ALGLIB --
	 Copyright 10.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpRandomizeFull(network *MultiLayerPerceptron) {
	mlpbase.MlpRandomizeFull(network.innerobj)
}

/*************************************************************************
Returns information about initialized network: number of inputs, outputs,
weights.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpProperties(network *MultiLayerPerceptron) (nin, nout, wcount int) {
	nin = 0
	nout = 0
	wcount = 0
	mlpbase.MlpProperties(network.innerobj, &nin, &nout, &wcount)
	return
}

/*************************************************************************
Tells whether network is SOFTMAX-normalized (i.e. classifier) or not.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpIsSoftMax(network *MultiLayerPerceptron) bool {
	return mlpbase.MlpIsSoftMax(network.innerobj)
}

/*************************************************************************
This function returns total number of layers (including input, hidden and
output layers).

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetLayersCount(network *MultiLayerPerceptron) int {
	return mlpbase.MlpGetLayersCount(network.innerobj)
}

/*************************************************************************
This function returns size of K-th layer.

K=0 corresponds to input layer, K=CNT-1 corresponds to output layer.

Size of the output layer is always equal to the number of outputs, although
when we have softmax-normalized network, last neuron doesn't have any
connections - it is just zero.

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetLayerSize(network *MultiLayerPerceptron, k int) (int, error) {
	return mlpbase.MlpGetLayerSize(network.innerobj, k)
}

/*************************************************************************
This function returns offset/scaling coefficients for I-th input of the
network.

INPUT PARAMETERS:
	Network     -   network
	I           -   input index

OUTPUT PARAMETERS:
	Mean        -   mean term
	Sigma       -   sigma term, guaranteed to be nonzero.

I-th input is passed through linear transformation
	IN[i] = (IN[i]-Mean)/Sigma
before feeding to the network

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetInputScaling(network *MultiLayerPerceptron, i int) (mean, sigma float64) {
	mean = 0.0
	sigma = 0.0
	mlpbase.MlpGetInputScaling(network.innerobj, i, &mean, &sigma)
	return
}

/*************************************************************************
This function returns offset/scaling coefficients for I-th output of the
network.

INPUT PARAMETERS:
	Network     -   network
	I           -   input index

OUTPUT PARAMETERS:
	Mean        -   mean term
	Sigma       -   sigma term, guaranteed to be nonzero.

I-th output is passed through linear transformation
	OUT[i] = OUT[i]*Sigma+Mean
before returning it to user. In case we have SOFTMAX-normalized network,
we return (Mean,Sigma)=(0.0,1.0).

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetOutputScaling(network *MultiLayerPerceptron, i int) (mean, sigma float64) {
	mean = 0.0
	sigma = 0.0
	mlpbase.MlpGetOutputScaling(network.innerobj, i, &mean, &sigma)
	return
}

/*************************************************************************
This function returns information about Ith neuron of Kth layer

INPUT PARAMETERS:
	Network     -   network
	K           -   layer index
	I           -   neuron index (within layer)

OUTPUT PARAMETERS:
	FKind       -   activation function type (used by MLPActivationFunction())
					this value is zero for input or linear neurons
	Threshold   -   also called offset, bias
					zero for input neurons

NOTE: this function throws exception if layer or neuron with  given  index
do not exists.

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetNeuronInfo(network *MultiLayerPerceptron, k, i int) (fkind int, threshold float64) {
	fkind = 0
	threshold = 0.0
	mlpbase.MlpGetNeuronInfo(network.innerobj, k, i, &fkind, &threshold)
	return
}

/*************************************************************************
This function returns information about connection from I0-th neuron of
K0-th layer to I1-th neuron of K1-th layer.

INPUT PARAMETERS:
	Network     -   network
	K0          -   layer index
	I0          -   neuron index (within layer)
	K1          -   layer index
	I1          -   neuron index (within layer)

RESULT:
	connection weight (zero for non-existent connections)

This function:
1. throws exception if layer or neuron with given index do not exists.
2. returns zero if neurons exist, but there is no connection between them

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetWeight(network *MultiLayerPerceptron, k0, i0, k1, i1 int) (float64, error) {
	return mlpbase.MlpGetWeight(network.innerobj, k0, i0, k1, i1)
}

/*************************************************************************
This function sets offset/scaling coefficients for I-th input of the
network.

INPUT PARAMETERS:
	Network     -   network
	I           -   input index
	Mean        -   mean term
	Sigma       -   sigma term (if zero, will be replaced by 1.0)

NTE: I-th input is passed through linear transformation
	IN[i] = (IN[i]-Mean)/Sigma
before feeding to the network. This function sets Mean and Sigma.

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpSetInputScaling(network *MultiLayerPerceptron, i int, mean, sigma float64) {
	mlpbase.MlpSetInputScaling(network.innerobj, i, mean, sigma)
}

/*************************************************************************
This function sets offset/scaling coefficients for I-th output of the
network.

INPUT PARAMETERS:
	Network     -   network
	I           -   input index
	Mean        -   mean term
	Sigma       -   sigma term (if zero, will be replaced by 1.0)

OUTPUT PARAMETERS:

NOTE: I-th output is passed through linear transformation
	OUT[i] = OUT[i]*Sigma+Mean
before returning it to user. This function sets Sigma/Mean. In case we
have SOFTMAX-normalized network, you can not set (Sigma,Mean) to anything
other than(0.0,1.0) - this function will throw exception.

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpSetOutputScaling(network *MultiLayerPerceptron, i int, mean, sigma float64) {
	mlpbase.MlpSetOutputScaling(network.innerobj, i, mean, sigma)
}

/*************************************************************************
This function modifies information about Ith neuron of Kth layer

INPUT PARAMETERS:
	Network     -   network
	K           -   layer index
	I           -   neuron index (within layer)
	FKind       -   activation function type (used by MLPActivationFunction())
					this value must be zero for input neurons
					(you can not set activation function for input neurons)
	Threshold   -   also called offset, bias
					this value must be zero for input neurons
					(you can not set threshold for input neurons)

NOTES:
1. this function throws exception if layer or neuron with given index do
   not exists.
2. this function also throws exception when you try to set non-linear
   activation function for input neurons (any kind of network) or for output
   neurons of classifier network.
3. this function throws exception when you try to set non-zero threshold for
   input neurons (any kind of network).

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpSetNeuronInfo(network *MultiLayerPerceptron, k, i, fkind int, threshold float64) {
	mlpbase.MlpSetNeuronInfo(network.innerobj, k, i, fkind, threshold)
}

/*************************************************************************
This function modifies information about connection from I0-th neuron of
K0-th layer to I1-th neuron of K1-th layer.

INPUT PARAMETERS:
	Network     -   network
	K0          -   layer index
	I0          -   neuron index (within layer)
	K1          -   layer index
	I1          -   neuron index (within layer)
	W           -   connection weight (must be zero for non-existent
					connections)

This function:
1. throws exception if layer or neuron with given index do not exists.
2. throws exception if you try to set non-zero weight for non-existent
   connection

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpSetWeight(network *MultiLayerPerceptron, k0, i0, k1, i1 int, w float64) {
	mlpbase.MlpSetWeight(network.innerobj, k0, i0, k1, i1, w)
}

/*************************************************************************
Neural network activation function

INPUT PARAMETERS:
	NET         -   neuron input
	K           -   function index (zero for linear function)

OUTPUT PARAMETERS:
	F           -   function
	DF          -   its derivative
	D2F         -   its second derivative

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpActivationFunction(net float64, k int) (f, df, d2f float64) {
	f = 0.0
	df = 0.0
	d2f = 0.0
	mlpbase.MlpActivationFunction(net, k, &f, &df, &d2f)
	return
}

/*************************************************************************
Procesing

INPUT PARAMETERS:
	Network -   neural network
	X       -   input vector,  array[0..NIn-1].

OUTPUT PARAMETERS:
	Y       -   result. Regression estimate when solving regression  task,
				vector of posterior probabilities for classification task.

See also MLPProcessI

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpProcess(network *MultiLayerPerceptron, x *[]float64) *[]float64 {
	y := make([]float64, 0)
	mlpbase.MlpProcess(network.innerobj, x, &y)
	return &y
}

/*************************************************************************
'interactive'  variant  of  MLPProcess  for  languages  like  Python which
support constructs like "Y = MLPProcess(NN,X)" and interactive mode of the
interpreter

This function allocates new array on each call,  so  it  is  significantly
slower than its 'non-interactive' counterpart, but it is  more  convenient
when you call it from command line.

  -- ALGLIB --
	 Copyright 21.09.2010 by Bochkanov Sergey
*************************************************************************/
func MlpProcessI(network *MultiLayerPerceptron, x *[]float64) *[]float64 {
	y := make([]float64, 0)
	mlpbase.MlpProcessi(network.innerobj, x, &y)
	return &y
}

/*************************************************************************
Error function for neural network, internal subroutine.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpError(network *MultiLayerPerceptron, xy *[][]float64, ssize  int) float64 {
	return mlpbase.MlpErrorN(network.innerobj, xy, ssize)
}

/*************************************************************************
Natural error function for neural network, internal subroutine.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpErrorN(network *MultiLayerPerceptron, xy *[][]float64, ssize int) float64 {
	return mlpbase.MlpErrorN(network.innerobj, xy, ssize)
}

/*************************************************************************
Classification error

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpClsError(network *MultiLayerPerceptron, xy *[][]float64, ssize int) int {
	return mlpbase.MlpClsError(network.innerobj, xy, ssize)
}

/*************************************************************************
Relative classification error on the test set

INPUT PARAMETERS:
	Network -   network
	XY      -   test set
	NPoints -   test set size

RESULT:
	percent of incorrectly classified cases. Works both for
	classifier networks and general purpose networks used as
	classifiers.

  -- ALGLIB --
	 Copyright 25.12.2008 by Bochkanov Sergey
*************************************************************************/
func MlpRelclsError(network *MultiLayerPerceptron, xy *[][]float64, npoints int) float64 {
	return mlpbase.MlpRelClsError(network.innerobj, xy, npoints)
}

/*************************************************************************
Average cross-entropy (in bits per element) on the test set

INPUT PARAMETERS:
	Network -   neural network
	XY      -   test set
	NPoints -   test set size

RESULT:
	CrossEntropy/(NPoints*LN(2)).
	Zero if network solves regression task.

  -- ALGLIB --
	 Copyright 08.01.2009 by Bochkanov Sergey
*************************************************************************/
func MlpAvgce(network *MultiLayerPerceptron, xy *[][]float64, npoints  int) float64 {
	return mlpbase.MlpAvgce(network.innerobj, xy, npoints)
}

/*************************************************************************
RMS error on the test set

INPUT PARAMETERS:
	Network -   neural network
	XY      -   test set
	NPoints -   test set size

RESULT:
	root mean square error.
	Its meaning for regression task is obvious. As for
	classification task, RMS error means error when estimating posterior
	probabilities.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpRmsError(network *MultiLayerPerceptron, xy *[][]float64, npoints int) float64 {
	return mlpbase.MlpRmsError(network.innerobj, xy, npoints)
}

/*************************************************************************
Average error on the test set

INPUT PARAMETERS:
	Network -   neural network
	XY      -   test set
	NPoints -   test set size

RESULT:
	Its meaning for regression task is obvious. As for
	classification task, it means average error when estimating posterior
	probabilities.

  -- ALGLIB --
	 Copyright 11.03.2008 by Bochkanov Sergey
*************************************************************************/
func mlpavgerror(network *MultiLayerPerceptron, xy *[][]float64, npoints int) float64 {
	return mlpbase.MlpAvgError(network.innerobj, xy, npoints)
}

/*************************************************************************
Average relative error on the test set

INPUT PARAMETERS:
	Network -   neural network
	XY      -   test set
	NPoints -   test set size

RESULT:
	Its meaning for regression task is obvious. As for
	classification task, it means average relative error when estimating
	posterior probability of belonging to the correct class.

  -- ALGLIB --
	 Copyright 11.03.2008 by Bochkanov Sergey
*************************************************************************/
func mlpavgrelerror(network *MultiLayerPerceptron, xy *[][]float64, npoints  int) float64 {
	return mlpbase.MlpAvgRelError(network.innerobj, xy, npoints)
}

/*************************************************************************
Gradient calculation

INPUT PARAMETERS:
	Network -   network initialized with one of the network creation funcs
	X       -   input vector, length of array must be at least NIn
	DesiredY-   desired outputs, length of array must be at least NOut
	Grad    -   possibly preallocated array. If size of array is smaller
				than WCount, it will be reallocated. It is recommended to
				reuse previously allocated array to reduce allocation
				overhead.

OUTPUT PARAMETERS:
	E       -   error function, SUM(sqr(y[i]-desiredy[i])/2,i)
	Grad    -   gradient of E with respect to weights of network, array[WCount]

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpGrad(network *MultiLayerPerceptron, x, desiredy, grad *[]float64) (e float64) {
	e = 0
	mlpbase.MlpGrad(network.innerobj, x, desiredy, &e, grad)
	return
}

/*************************************************************************
Gradient calculation (natural error function is used)

INPUT PARAMETERS:
	Network -   network initialized with one of the network creation funcs
	X       -   input vector, length of array must be at least NIn
	DesiredY-   desired outputs, length of array must be at least NOut
	Grad    -   possibly preallocated array. If size of array is smaller
				than WCount, it will be reallocated. It is recommended to
				reuse previously allocated array to reduce allocation
				overhead.

OUTPUT PARAMETERS:
	E       -   error function, sum-of-squares for regression networks,
				cross-entropy for classification networks.
	Grad    -   gradient of E with respect to weights of network, array[WCount]

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpGradN(network *MultiLayerPerceptron, x, desiredy, grad *[]float64) (e float64) {
	e = 0
	mlpbase.MlpGradn(network.innerobj, x, desiredy, &e, grad)
	return
}

/*************************************************************************
Batch gradient calculation for a set of inputs/outputs

INPUT PARAMETERS:
	Network -   network initialized with one of the network creation funcs
	XY      -   set of inputs/outputs; one sample = one row;
				first NIn columns contain inputs,
				next NOut columns - desired outputs.
	SSize   -   number of elements in XY
	Grad    -   possibly preallocated array. If size of array is smaller
				than WCount, it will be reallocated. It is recommended to
				reuse previously allocated array to reduce allocation
				overhead.

OUTPUT PARAMETERS:
	E       -   error function, SUM(sqr(y[i]-desiredy[i])/2,i)
	Grad    -   gradient of E with respect to weights of network, array[WCount]

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpGradBatch(network *MultiLayerPerceptron, xy *[][]float64, ssize int, grad *[]float64) (e float64) {
	e = 0
	mlpbase.MlpGradBatch(network.innerobj, xy, ssize, &e, grad)
	return
}

/*************************************************************************
Batch gradient calculation for a set of inputs/outputs
(natural error function is used)

INPUT PARAMETERS:
	Network -   network initialized with one of the network creation funcs
	XY      -   set of inputs/outputs; one sample = one row;
				first NIn columns contain inputs,
				next NOut columns - desired outputs.
	SSize   -   number of elements in XY
	Grad    -   possibly preallocated array. If size of array is smaller
				than WCount, it will be reallocated. It is recommended to
				reuse previously allocated array to reduce allocation
				overhead.

OUTPUT PARAMETERS:
	E       -   error function, sum-of-squares for regression networks,
				cross-entropy for classification networks.
	Grad    -   gradient of E with respect to weights of network, array[WCount]

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpGradNBatch(network *MultiLayerPerceptron, xy *[][]float64, ssize int, grad *[]float64) (e float64) {
	e = 0
	mlpbase.MlpGradNBatch(network.innerobj, xy, ssize, &e, grad)
	return
}

/*************************************************************************
Batch Hessian calculation (natural error function) using R-algorithm.
Internal subroutine.

  -- ALGLIB --
	 Copyright 26.01.2008 by Bochkanov Sergey.

	 Hessian calculation based on R-algorithm described in
	 "Fast Exact Multiplication by the Hessian",
	 B. A. Pearlmutter,
	 Neural Computation, 1994.
*************************************************************************/
func MlpHessianNBatch(network *MultiLayerPerceptron, xy *[][]float64, ssize int, e *float64, grad *[]float64, h *[][]float64) {
	mlpbase.MlpHessianNBatch(network.innerobj, xy, ssize, e, grad, h)
}

/*************************************************************************
Batch Hessian calculation using R-algorithm.
Internal subroutine.

  -- ALGLIB --
	 Copyright 26.01.2008 by Bochkanov Sergey.

	 Hessian calculation based on R-algorithm described in
	 "Fast Exact Multiplication by the Hessian",
	 B. A. Pearlmutter,
	 Neural Computation, 1994.
*************************************************************************/
func MlpHessianBatch(network *MultiLayerPerceptron, xy *[][]float64, ssize int, e *float64, grad *[]float64, h *[][]float64) {
	mlpbase.MlpHessianBatch(network.innerobj, xy, ssize, e, grad, h)
}