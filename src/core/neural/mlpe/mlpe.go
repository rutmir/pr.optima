package mlpe

import (
	"pr.optima/src/core/neural/mlpbase"
	"fmt"
	"math/rand"
	"math"
	"core/neural/mlptrain"
	"pr.optima/src/core/neural/mlptrain"
)

const (
	mlpntotaloffset = 3
	mlpevnum = 9

	maxrealnumber = 1E300
	minrealnumber = 1E-300
)

type mlpensemble struct
{
	structinfo     [...]int
	ensemblesize   int
	nin            int
	nout           int
	wcount         int
	issoftmax      bool
	postprocessing bool
	weights        [...]float64
	columnmeans    [...]float64
	columnsigmas   [...]float64
	serializedlen  int
	serializedmlp  [...]float64
	tmpweights     [...]float64
	tmpmeans       [...]float64
	tmpsigmas      [...]float64
	neurons        [...]float64
	dfdnet         [...]float64
	y              [...]float64
}

func NewMpl() *mlpensemble {
	return &mlpensemble{
		structinfo     :[0]int,
		weights        :[0]float64,
		columnmeans    :[0]float64,
		columnsigmas   :[0]float64,
		serializedmlp  :[0]float64,
		tmpweights     :[0]float64,
		tmpmeans       :[0]float64,
		tmpsigmas      :[0]float64,
		neurons        :[0]float64,
		dfdnet         :[0]float64,
		y              :[0]float64}
}


/*************************************************************************
Like MLPCreate0, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreate0(nin, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreate0(nin, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreate1, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreate1(nin, nhid, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreate1(nin, nhid, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreate2, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreate2(nin, nhid1, nhid2, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreate2(nin, nhid1, nhid2, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateB0, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateB0(nin, nout int, b, d float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreateb0(nin, nout, b, d, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateB1, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateB1(nin, nhid, nout int, b, d float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreateb1(nin, nhid, nout, b, d, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateB2, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateB2(nin, nhid1, nhid2, nout int, b, d float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreateb2(nin, nhid1, nhid2, nout, b, d, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateR0, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateR0(nin, nout int, a, b float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreater0(nin, nout, a, b, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateR1, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateR1(nin, nhid, nout int, a, b float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreater1(nin, nhid, nout, a, b, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateR2, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateR2(nin, nhid1, nhid2, nout int, a, b float64, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreater2(nin, nhid1, nhid2, nout, a, b, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateC0, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateC0(nin, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreatec0(nin, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateC1, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateC1(nin, nhid, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreatec1(nin, nhid, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Like MLPCreateC2, but for ensembles.

  -- ALGLIB --
	 Copyright 18.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreateC2(nin, nhid1, nhid2, nout, ensemblesize int, ensemble *mlpensemble) error {
	net := mlpbase.NewMlp()

	if err := mlpbase.MlpCreatec2(nin, nhid1, nhid2, nout, net); err != nil {
		return err
	}
	return MlpeCreatefromnetwork(net, ensemblesize, ensemble)
}

/*************************************************************************
Creates ensemble from network. Only network geometry is copied.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCreatefromnetwork(network *mlpbase.Multilayerperceptron, ensemblesize int, ensemble *mlpensemble) error {
	i := 0
	ccount := 0
	i_ := 0
	i1_ := 0


	if !(ensemblesize > 0) {
		return fmt.Errorf("MLPECreate: incorrect ensemble size!")
	}

	//
	// network properties
	//
	mlpbase.MlpProperties(network, &ensemble.nin, &ensemble.nout, &ensemble.wcount)
	if mlpbase.MlpIsSoftMax(network) {
		ccount = ensemble.nin
	}else {
		ccount = ensemble.nin + ensemble.nout
	}
	ensemble.postprocessing = false
	ensemble.issoftmax = mlpbase.MlpIsSoftMax(network)
	ensemble.ensemblesize = ensemblesize

	//
	// structure information
	//
	ensemble.structinfo = [network.StructInfo[0] - 1 + 1]int
	for i = 0; i <= network.StructInfo[0] - 1; i++ {
		ensemble.structinfo[i] = network.StructInfo[i]
	}

	//
	// weights, means, sigmas
	//
	ensemble.weights = [ensemblesize * ensemble.wcount - 1 + 1]float64
	ensemble.columnmeans = [ensemblesize * ccount - 1 + 1]float64
	ensemble.columnsigmas = [ensemblesize * ccount - 1 + 1]float64
	for i = 0; i <= ensemblesize * ensemble.wcount - 1; i++ {
		ensemble.weights[i] = rand.Float64() - 0.5
	}
	for i = 0; i <= ensemblesize - 1; i++ {
		i1_ = (0) - (i * ccount)
		for i_ = i * ccount; i_ <= (i + 1) * ccount - 1; i_++ {
			ensemble.columnmeans[i_] = network.ColumnMeans[i_ + i1_]
		}
		i1_ = (0) - (i * ccount)
		for i_ = i * ccount; i_ <= (i + 1) * ccount - 1; i_++ {
			ensemble.columnsigmas[i_] = network.ColumnSigmas[i_ + i1_]
		}
	}

	//
	// serialized part
	//
	mlpbase.MlpSerializeOld(network, &ensemble.serializedmlp, &ensemble.serializedlen)

	//
	// temporaries, internal buffers
	//
	ensemble.tmpweights = [ensemble.wcount - 1 + 1] float64
	ensemble.tmpmeans = [ccount - 1 + 1]float64
	ensemble.tmpsigmas = [ccount - 1 + 1]float64
	ensemble.neurons = [ensemble.structinfo[mlpntotaloffset] - 1 + 1]float64
	ensemble.dfdnet = [ensemble.structinfo[mlpntotaloffset] - 1 + 1]float64
	ensemble.y = [ensemble.nout - 1 + 1]float64

	return nil
}

/*************************************************************************
Copying of MLPEnsemble strucure

INPUT PARAMETERS:
	Ensemble1 -   original

OUTPUT PARAMETERS:
	Ensemble2 -   copy

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeCopy(ensemble1 *mlpensemble, ensemble2 *mlpensemble) {
	ccount := 0

	//
	// Unload info
	//
	ssize := ensemble1.structinfo[0]
	if ensemble1.issoftmax {
		ccount = ensemble1.nin
	}else {
		ccount = ensemble1.nin + ensemble1.nout
	}
	ntotal := ensemble1.structinfo[mlpntotaloffset]

	//
	// Allocate space
	//
	ensemble2.structinfo = [ssize - 1 + 1]int
	ensemble2.weights = [ensemble1.ensemblesize * ensemble1.wcount - 1 + 1]float64
	ensemble2.columnmeans = [ensemble1.ensemblesize * ccount - 1 + 1]float64
	ensemble2.columnsigmas = [ensemble1.ensemblesize * ccount - 1 + 1]float64
	ensemble2.tmpweights = [ensemble1.wcount - 1 + 1]float64
	ensemble2.tmpmeans = [ccount - 1 + 1]float64
	ensemble2.tmpsigmas = [ccount - 1 + 1]float64
	ensemble2.serializedmlp = [ensemble1.serializedlen - 1 + 1]float64
	ensemble2.neurons = [ntotal - 1 + 1]float64
	ensemble2.dfdnet = [ntotal - 1 + 1]float64
	ensemble2.y = [ensemble1.nout - 1 + 1]float64

	//
	// Copy
	//
	ensemble2.nin = ensemble1.nin
	ensemble2.nout = ensemble1.nout
	ensemble2.wcount = ensemble1.wcount
	ensemble2.ensemblesize = ensemble1.ensemblesize
	ensemble2.issoftmax = ensemble1.issoftmax
	ensemble2.postprocessing = ensemble1.postprocessing
	ensemble2.serializedlen = ensemble1.serializedlen

	for i := 0; i <= ssize - 1; i++ {
		ensemble2.structinfo[i] = ensemble1.structinfo[i]
	}
	for i := 0; i <= ensemble1.ensemblesize * ensemble1.wcount - 1; i++ {
		ensemble2.weights[i] = ensemble1.weights[i]
	}
	for i := 0; i <= ensemble1.ensemblesize * ccount - 1; i++ {
		ensemble2.columnmeans[i] = ensemble1.columnmeans[i]
	}
	for i := 0; i <= ensemble1.ensemblesize * ccount - 1; i++ {
		ensemble2.columnsigmas[i] = ensemble1.columnsigmas[i]
	}
	for i := 0; i <= ensemble1.serializedlen - 1; i++ {
		ensemble2.serializedmlp[i] = ensemble1.serializedmlp[i]
	}
}

/*************************************************************************
Serialization of MLPEnsemble strucure

INPUT PARAMETERS:
	Ensemble-   original

OUTPUT PARAMETERS:
	RA      -   array of real numbers which stores ensemble,
				array[0..RLen-1]
	RLen    -   RA lenght

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeSerialize(ensemble *mlpensemble, ra *[]float64, rlen *int) {
	i := 0
	ssize := 0
	ntotal := 0
	ccount := 0
	hsize := 0
	offs := 0
	i_ := 0
	i1_ := 0

	rlen = 0

	hsize = 13;
	ssize = ensemble.structinfo[0]
	if ensemble.issoftmax {
		ccount = ensemble.nin
	}else {
		ccount = ensemble.nin + ensemble.nout
	}
	ntotal = ensemble.structinfo[mlpntotaloffset]
	rlen = hsize + ssize + ensemble.ensemblesize * ensemble.wcount + 2 * ccount * ensemble.ensemblesize + ensemble.serializedlen

	//
	//  RA format:
	//  [0]     RLen
	//  [1]     Version (MLPEVNum)
	//  [2]     EnsembleSize
	//  [3]     NIn
	//  [4]     NOut
	//  [5]     WCount
	//  [6]     IsSoftmax 0/1
	//  [7]     PostProcessing 0/1
	//  [8]     sizeof(StructInfo)
	//  [9]     NTotal (sizeof(Neurons), sizeof(DFDNET))
	//  [10]    CCount (sizeof(ColumnMeans), sizeof(ColumnSigmas))
	//  [11]    data offset
	//  [12]    SerializedLen
	//
	//  [..]    StructInfo
	//  [..]    Weights
	//  [..]    ColumnMeans
	//  [..]    ColumnSigmas
	//
	ra = [rlen - 1 + 1]float64
	ra[0] = rlen
	ra[1] = mlpevnum
	ra[2] = ensemble.ensemblesize
	ra[3] = ensemble.nin
	ra[4] = ensemble.nout
	ra[5] = ensemble.wcount

	if ensemble.issoftmax {
		ra[6] = 1
	}else {
		ra[6] = 0
	}
	if ensemble.postprocessing {
		ra[7] = 1
	}else {
		ra[7] = 9
	}
	ra[8] = ssize
	ra[9] = ntotal
	ra[10] = ccount
	ra[11] = hsize
	ra[12] = ensemble.serializedlen
	offs = hsize
	for i = offs; i <= offs + ssize - 1; i++ {
		ra[i] = ensemble.structinfo[i - offs]
	}
	offs = offs + ssize
	i1_ = (0) - (offs)
	for i_ = offs; i_ <= offs + ensemble.ensemblesize * ensemble.wcount - 1; i_++ {
		ra[i_] = ensemble.weights[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ensemble.wcount
	i1_ = (0) - (offs)
	for i_ = offs; i_ <= offs + ensemble.ensemblesize * ccount - 1; i_++ {
		ra[i_] = ensemble.columnmeans[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ccount
	i1_ = (0) - (offs)
	for i_ = offs; i_ <= offs + ensemble.ensemblesize * ccount - 1; i_++ {
		ra[i_] = ensemble.columnsigmas[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ccount
	i1_ = (0) - (offs)
	for i_ = offs; i_ <= offs + ensemble.serializedlen - 1; i_++ {
		ra[i_] = ensemble.serializedmlp[i_ + i1_]
	}
	offs = offs + ensemble.serializedlen
}

/*************************************************************************
Unserialization of MLPEnsemble strucure

INPUT PARAMETERS:
	RA      -   real array which stores ensemble

OUTPUT PARAMETERS:
	Ensemble-   restored structure

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeUnserialize(ra *[]float64, ensemble *mlpensemble) {
	i := 0
	ssize := 0
	ntotal := 0
	ccount := 0
	offs := 0
	i_ := 0
	i1_ := 0

	if !(int(round(ra[1])) == mlpevnum) {
		return fmt.Errorf("MLPEUnserialize: incorrect array!")
	}

	//
	// load info
	//
	ensemble.ensemblesize = int(round(ra[2]))
	ensemble.nin = int(round(ra[3]))
	ensemble.nout = int(round(ra[4]))
	ensemble.wcount = int(round(ra[5]))
	ensemble.issoftmax = int(round(ra[6])) == 1
	ensemble.postprocessing = int(round(ra[7])) == 1
	ssize = int(round(ra[8]))
	ntotal = int(round(ra[9]))
	ccount = int(round(ra[10]))
	offs = int(round(ra[11]))
	ensemble.serializedlen = int(round(ra[12]))

	//
	//  Allocate arrays
	//
	ensemble.structinfo = [ssize - 1 + 1]int
	ensemble.weights = [ensemble.ensemblesize * ensemble.wcount - 1 + 1]float64
	ensemble.columnmeans = [ensemble.ensemblesize * ccount - 1 + 1]float64
	ensemble.columnsigmas = [ensemble.ensemblesize * ccount - 1 + 1]float64
	ensemble.tmpweights = [ensemble.wcount - 1 + 1]float64
	ensemble.tmpmeans = [ccount - 1 + 1]float64
	ensemble.tmpsigmas = [ccount - 1 + 1]float64
	ensemble.neurons = [ntotal - 1 + 1]float64
	ensemble.dfdnet = [ntotal - 1 + 1]float64
	ensemble.serializedmlp = [ensemble.serializedlen - 1 + 1]float64
	ensemble.y = [ensemble.nout - 1 + 1]float64

	//
	// load data
	//
	for i = offs; i <= offs + ssize - 1; i++ {
		ensemble.structinfo[i - offs] = int(round(ra[i]))
	}
	offs = offs + ssize
	i1_ = (offs) - (0)
	for i_ = 0; i_ <= ensemble.ensemblesize * ensemble.wcount - 1; i_++ {
		ensemble.weights[i_] = ra[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ensemble.wcount
	i1_ = (offs) - (0)
	for i_ = 0; i_ <= ensemble.ensemblesize * ccount - 1; i_++ {
		ensemble.columnmeans[i_] = ra[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ccount
	i1_ = (offs) - (0)
	for i_ = 0; i_ <= ensemble.ensemblesize * ccount - 1; i_++ {
		ensemble.columnsigmas[i_] = ra[i_ + i1_]
	}
	offs = offs + ensemble.ensemblesize * ccount
	i1_ = (offs) - (0)
	for i_ = 0; i_ <= ensemble.serializedlen - 1; i_++ {
		ensemble.serializedmlp[i_] = ra[i_ + i1_]
	}
	offs = offs + ensemble.serializedlen
	return nil
}

/*************************************************************************
Randomization of MLP ensemble

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeRandomize(ensemble *mlpensemble) {
	for i := 0; i <= ensemble.ensemblesize * ensemble.wcount - 1; i++ {
		ensemble.weights[i] = rand.Float64() - 0.5
	}
}

/*************************************************************************
Return ensemble properties (number of inputs and outputs).

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeProperties(ensemble *mlpensemble, nin, nout *int) {
	nin = ensemble.nin
	nout = ensemble.nout
}

/*************************************************************************
Return normalization type (whether ensemble is SOFTMAX-normalized or not).

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeIsSoftMax(ensemble *mlpensemble) bool {
	return ensemble.issoftmax
}

/*************************************************************************
Procesing

INPUT PARAMETERS:
	Ensemble-   neural networks ensemble
	X       -   input vector,  array[0..NIn-1].
	Y       -   (possibly) preallocated buffer; if size of Y is less than
				NOut, it will be reallocated. If it is large enough, it
				is NOT reallocated, so we can save some time on reallocation.


OUTPUT PARAMETERS:
	Y       -   result. Regression estimate when solving regression  task,
				vector of posterior probabilities for classification task.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeProcess(ensemble *mlpensemble, x, y *[]float64) error {
	i := 0
	cc := 0
	i_ := 0
	i1_ := 0

	if len(y) < ensemble.nout {
		y = [ensemble.nout]float64
	}
	es := ensemble.ensemblesize
	wc := ensemble.wcount
	if ensemble.issoftmax {
		cc = ensemble.nin
	}else {
		cc = ensemble.nin + ensemble.nout
	}
	v := 1 / float64(es)
	for i = 0; i <= ensemble.nout - 1; i++ {
		y[i] = 0
	}
	for i = 0; i <= es - 1; i++ {
		i1_ = (i * wc) - (0)
		for i_ = 0; i_ <= wc - 1; i_++ {
			ensemble.tmpweights[i_] = ensemble.weights[i_ + i1_]
		}
		i1_ = (i * cc) - (0)
		for i_ = 0; i_ <= cc - 1; i_++ {
			ensemble.tmpmeans[i_] = ensemble.columnmeans[i_ + i1_]
		}
		i1_ = (i * cc) - (0)
		for i_ = 0; i_ <= cc - 1; i_++ {
			ensemble.tmpsigmas[i_] = ensemble.columnsigmas[i_ + i1_]
		}
		if err := mlpbase.MlpInternalProcessVector(&ensemble.structinfo, &ensemble.tmpweights, &ensemble.tmpmeans, &ensemble.tmpsigmas, &ensemble.neurons, &ensemble.dfdnet, x, &ensemble.y); err != nil {
			return err
		}
		for i_ = 0; i_ <= ensemble.nout - 1; i_++ {
			y[i_] = y[i_] + v * ensemble.y[i_]
		}
	}
	return nil
}

/*************************************************************************
'interactive'  variant  of  MLPEProcess  for  languages  like Python which
support constructs like "Y = MLPEProcess(LM,X)" and interactive mode of the
interpreter

This function allocates new array on each call,  so  it  is  significantly
slower than its 'non-interactive' counterpart, but it is  more  convenient
when you call it from command line.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeProcessI(ensemble *mlpensemble, x, y *[]float64) error {
	y = [0]float64
	return MlpeProcess(ensemble, x, y)
}

/*************************************************************************
Relative classification error on the test set

INPUT PARAMETERS:
	Ensemble-   ensemble
	XY      -   test set
	NPoints -   test set size

RESULT:
	percent of incorrectly classified cases.
	Works both for classifier betwork and for regression networks which
are used as classifiers.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeRelclsError(ensemble *mlpensemble, xy *[][]float64, npoints int) (float64 error) {
	relcls := 0.0
	avgce := 0.0
	rms := 0.0
	avg := 0.0
	avgrel := 0.0

	err := mlpeallerrors(ensemble, xy, npoints, &relcls, &avgce, &rms, &avg, &avgrel)
	return relcls, err
}

/*************************************************************************
Average cross-entropy (in bits per element) on the test set

INPUT PARAMETERS:
	Ensemble-   ensemble
	XY      -   test set
	NPoints -   test set size

RESULT:
	CrossEntropy/(NPoints*LN(2)).
	Zero if ensemble solves regression task.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeAvgce(ensemble *mlpensemble, xy *[][]float64, npoints int) (float64, error) {
	relcls := 0.0
	avgce := 0.0
	rms := 0.0
	avg := 0.0
	avgrel := 0.0

	err := mlpeallerrors(ensemble, xy, npoints, &relcls, &avgce, &rms, &avg, &avgrel)
	return avgce, err
}

/*************************************************************************
RMS error on the test set

INPUT PARAMETERS:
	Ensemble-   ensemble
	XY      -   test set
	NPoints -   test set size

RESULT:
	root mean square error.
	Its meaning for regression task is obvious. As for classification task
RMS error means error when estimating posterior probabilities.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeRmsError(ensemble *mlpensemble, xy *[][]float64, npoints int) (float64, error) {
	relcls := 0.0
	avgce := 0.0
	rms := 0.0
	avg := 0.0
	avgrel := 0.0

	err := mlpeallerrors(ensemble, xy, npoints, &relcls, &avgce, &rms, &avg, &avgrel)
	return rms, err
}

/*************************************************************************
Average relative error on the test set

INPUT PARAMETERS:
	Ensemble-   ensemble
	XY      -   test set
	NPoints -   test set size

RESULT:
	Its meaning for regression task is obvious. As for classification task
it means average relative error when estimating posterior probabilities.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeAvgrelError(ensemble *mlpensemble, xy *[][]float64, npoints int) (float64, error) {
	relcls := 0.0
	avgce := 0.0
	rms := 0.0
	avg := 0.0
	avgrel := 0.0

	err := mlpeallerrors(ensemble, xy, npoints, &relcls, &avgce, &rms, &avg, &avgrel)
	return avgrel, err
}

/*************************************************************************
Average error on the test set

INPUT PARAMETERS:
	Ensemble-   ensemble
	XY      -   test set
	NPoints -   test set size

RESULT:
	Its meaning for regression task is obvious. As for classification task
it means average error when estimating posterior probabilities.

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func MlpeAvgError(ensemble *mlpensemble, xy *[][]float64, npoints int) (float64, error) {
	relcls := 0.0
	avgce := 0.0
	rms := 0.0
	avg := 0.0
	avgrel := 0.0

	err := mlpeallerrors(ensemble, xy, npoints, &relcls, &avgce, &rms, &avg, &avgrel)
	return avg, err
}

/*************************************************************************
Calculation of all types of errors

  -- ALGLIB --
	 Copyright 17.02.2009 by Bochkanov Sergey
*************************************************************************/
func mlpeallerrors(ensemble *mlpensemble, xy *[][]float64, npoints int, relcls, avgce, rms, avg, avgrel *float64) error {
	i := 0
	buf := [0]float64
	dy := [0]float64
	i_ := 0
	i1_ := 0

	relcls = 0.0
	avgce = 0.0
	rms = 0.0
	avg = 0.0
	avgrel = 0.0

	workx := [ensemble.nin - 1 + 1]float64
	y := [ensemble.nout - 1 + 1]float64
	if ensemble.issoftmax {
		dy = [0 + 1]float64
		dserrallocate(ensemble.nout, &buf)
	}else {
		dy = [ensemble.nout - 1 + 1]float64
		dserrallocate(-ensemble.nout, &buf)
	}
	for i = 0; i <= npoints - 1; i++ {
		for i_ = 0; i_ <= ensemble.nin - 1; i_++ {
			workx[i_] = xy[i][ i_]
		}
		if err := MlpeProcess(ensemble, &workx, &y); err != nil {
			return err
		}
		if ensemble.issoftmax {
			dy[0] = xy[i][ ensemble.nin]
		}else {
			i1_ = (ensemble.nin) - (0)
			for i_ = 0; i_ <= ensemble.nout - 1; i_++ {
				dy[i_] = xy[i][ i_ + i1_]
			}
		}
		dserraccumulate(&buf, &y, &dy)
	}
	dserrfinish(&buf)
	relcls = buf[0]
	avgce = buf[1]
	rms = buf[2]
	avg = buf[3]
	avgrel = buf[4]
	return nil
}

/*************************************************************************
Internal bagging subroutine.

  -- ALGLIB --
	 Copyright 19.02.2009 by Bochkanov Sergey
*************************************************************************/
func mlpebagginginternal(
ensemble *mlpensemble,
xy *[][]float64,
npoints int, decay float64,
restarts int, wstep float64, maxits int, lmalgorithm bool, info *int, rep *mlptrain.MlpReport, ooberrors *mlptrain.MlpCvReport) {
	xys := [0][0]float64
	s := [0]bool
	oobbuf := [0][0]float64
	oobcntbuf := [0]int
	x := [0]float64
	y := [0]float64
	dy := [0]float64
	dsbuf := [0]float64
	nin := 0
	nout := 0
	ccnt := 0
	pcnt := 0
	i := 0
	j := 0
	k := 0
	v := 0.0
	tmprep := &mlptrain.MlpReport{}
	network := mlpbase.NewMlp()
	i_ := 0
	i1_ := 0

	info = 0;


	//
	// Test for inputs
	//
	if (!lmalgorithm & wstep == 0) & maxits == 0 {
		info = -8
		return
	}
	if ((npoints <= 0 | restarts < 1) | wstep < 0) | maxits < 0 {
		info = -1
		return
	}
	if ensemble.issoftmax {
		for i = 0; i <= npoints - 1; i++ {
			if int(round(xy[i][ ensemble.nin])) < 0 | int(round(xy[i][ ensemble.nin])) >= ensemble.nout {
				info = -2
				return
			}
		}
	}

	//
	// allocate temporaries
	//
	info = 2
	rep.NGrad = 0
	rep.NHess = 0
	rep.NCholesky = 0
	ooberrors.RelclsError = 0
	ooberrors.Avgce = 0
	ooberrors.RmsError = 0
	ooberrors.AvgError = 0
	ooberrors.AvgrelError = 0
	nin = ensemble.nin
	nout = ensemble.nout
	if ensemble.issoftmax {
		ccnt = nin + 1
		pcnt = nin
	}else {
		ccnt = nin + nout
		pcnt = nin + nout
	}
	xys = [npoints - 1 + 1][ccnt - 1 + 1]float64
	s = [npoints - 1 + 1]bool
	oobbuf = [npoints - 1 + 1][ nout - 1 + 1]float64
	oobcntbuf = [npoints - 1 + 1]int
	x = [nin - 1 + 1]float64
	y = [nout - 1 + 1]float64
	if ensemble.issoftmax {
		dy = [0 + 1]float64
	}else {
		dy = [nout - 1 + 1]float64
	}
	for i = 0; i <= npoints - 1; i++ {
		for j = 0; j <= nout - 1; j++ {
			oobbuf[i][ j] = 0
		}
	}
	for i = 0; i <= npoints - 1; i++ {
		oobcntbuf[i] = 0
	}
	mlpbase.MlpUnserializeOld(ensemble.serializedmlp, network)

	//
	// main bagging cycle
	//
	for k = 0; k <= ensemble.ensemblesize - 1; k++ {
		//
		// prepare dataset
		//
		for i = 0; i <= npoints - 1; i++ {
			s[i] = false
		}
		for i = 0; i <= npoints - 1; i++ {
			j = rand.Intn(npoints)
			s[j] = true
			for i_ = 0; i_ <= ccnt - 1; i_++ {
				xys[i][ i_] = xy[j][ i_]
			}
		}

		//
		// train
		//
		if lmalgorithm {
			mlptrain.mlptrainlm(network, xys, npoints, decay, restarts, ref info, tmprep)
		}else {
			mlptrain.mlptrainlbfgs(network, xys, npoints, decay, restarts, wstep, maxits, ref info, tmprep)
		}
		if info < 0 {
			return
		}

		//
		// save results
		//
		rep.NGrad += tmprep.NGrad
		rep.NHess += tmprep.NHess
		rep.NCholesky += tmprep.NCholesky
		i1_ = (0) - (k * ensemble.wcount)
		for i_ = k * ensemble.wcount; i_ <= (k + 1) * ensemble.wcount - 1; i_++ {
			ensemble.weights[i_] = network.Weights[i_ + i1_]
		}
		i1_ = (0) - (k * pcnt)
		for i_ = k * pcnt; i_ <= (k + 1) * pcnt - 1; i_++ {
			ensemble.columnmeans[i_] = network.ColumnMeans[i_ + i1_]
		}
		i1_ = (0) - (k * pcnt)
		for i_ = k * pcnt; i_ <= (k + 1) * pcnt - 1; i_++ {
			ensemble.columnsigmas[i_] = network.ColumnSigmas[i_ + i1_]
		}

		//
		// OOB estimates
		//
		for i = 0; i <= npoints - 1; i++ {
			if !s[i] {
				for i_ = 0; i_ <= nin - 1; i_++ {
					x[i_] = xy[i][ i_]
				}
				mlpbase.MlpProcess(network, &x, &y)
				for i_ = 0; i_ <= nout - 1; i_++ {
					oobbuf[i][ i_] = oobbuf[i][ i_] + y[i_]
				}
				oobcntbuf[i] = oobcntbuf[i] + 1
			}
		}
	}

	//
	// OOB estimates
	//
	if ensemble.issoftmax {
		dserrallocate(nout, &dsbuf)
	}else {
		dserrallocate(-nout, &dsbuf)
	}
	for i = 0; i <= npoints - 1; i++ {
		if oobcntbuf[i] != 0 {
			v = 1 / float64(oobcntbuf[i])
			for i_ = 0; i_ <= nout - 1; i_++ {
				y[i_] = v * oobbuf[i][ i_]
			}
			if ensemble.issoftmax {
				dy[0] = xy[i][ nin]
			}else {
				i1_ = (nin) - (0)
				for i_ = 0; i_ <= nout - 1; i_++ {
					dy[i_] = v * xy[i][ i_ + i1_]
				}
			}
			dserraccumulate(&dsbuf, y, dy)
		}
	}
	dserrfinish(&dsbuf)
	ooberrors.RelclsError = dsbuf[0]
	ooberrors.Avgce = dsbuf[1]
	ooberrors.RmsError = dsbuf[2]
	ooberrors.AvgError = dsbuf[3]
	ooberrors.AvgrelError = dsbuf[4]
	ooberrors.AvgrelError = dsbuf[4]
}

func round(f float64) float64 {
	return math.Floor(f + .5)
}

/*************************************************************************
This set of routines (DSErrAllocate, DSErrAccumulate, DSErrFinish)
calculates different error functions (classification error, cross-entropy,
rms, avg, avg.rel errors).

1. DSErrAllocate prepares buffer.
2. DSErrAccumulate accumulates individual errors:
	* Y contains predicted output (posterior probabilities for classification)
	* DesiredY contains desired output (class number for classification)
3. DSErrFinish outputs results:
   * Buf[0] contains relative classification error (zero for regression tasks)
   * Buf[1] contains avg. cross-entropy (zero for regression tasks)
   * Buf[2] contains rms error (regression, classification)
   * Buf[3] contains average error (regression, classification)
   * Buf[4] contains average relative error (regression, classification)

NOTES(1):
	"NClasses>0" means that we have classification task.
	"NClasses<0" means regression task with -NClasses real outputs.

NOTES(2):
	rms. avg, avg.rel errors for classification tasks are interpreted as
	errors in posterior probabilities with respect to probabilities given
	by training/test set.

  -- ALGLIB --
	 Copyright 11.01.2009 by Bochkanov Sergey
*************************************************************************/
func dserrallocate(nclasses int, buf *[]float64) {
	buf = [7 + 1]float64
	buf[0] = 0
	buf[1] = 0
	buf[2] = 0
	buf[3] = 0
	buf[4] = 0
	buf[5] = nclasses
	buf[6] = 0
	buf[7] = 0
}

/*************************************************************************
See DSErrAllocate for comments on this routine.

  -- ALGLIB --
	 Copyright 11.01.2009 by Bochkanov Sergey
*************************************************************************/
func dserraccumulate(buf, y, desiredy *[]float64) {
	nclasses := 0
	nout := 0
	mmax := 0
	rmax := 0
	j := 0
	v := 0.0
	ev := 0.0

	offs := 5
	nclasses = int(round(buf[offs]))
	if nclasses > 0 {
		//
		// Classification
		//
		rmax = int(round(desiredy[0]))
		mmax = 0
		for j = 1; j <= nclasses - 1; j++ {
			if y[j] > y[mmax] {
				mmax = j
			}
		}
		if mmax != rmax {
			buf[0] = buf[0] + 1
		}
		if y[rmax] > 0 {
			buf[1] = buf[1] - math.Log(y[rmax])
		}else {
			buf[1] = buf[1] + math.Log(maxrealnumber)
		}
		for j = 0; j <= nclasses - 1; j++ {
			v = y[j]
			if j == rmax {
				ev = 1
			}else {
				ev = 0
			}
			v2 := (v - ev) * (v - ev)
			buf[2] = buf[2] + v2
			buf[3] = buf[3] + math.Abs(v - ev)
			if ev != 0 {
				buf[4] = buf[4] + math.Abs((v - ev) / ev)
				buf[offs + 2] = buf[offs + 2] + 1
			}
		}
		buf[offs + 1] = buf[offs + 1] + 1
	}else {
		//
		// Regression
		//
		nout = -nclasses
		rmax = 0
		for j = 1; j <= nout - 1; j++ {
			if desiredy[j] > desiredy[rmax] {
				rmax = j
			}
		}
		mmax = 0
		for j = 1; j <= nout - 1; j++ {
			if y[j] > y[mmax] {
				mmax = j
			}
		}
		if mmax != rmax {
			buf[0] = buf[0] + 1
		}
		for j = 0; j <= nout - 1; j++ {
			v = y[j]
			ev = desiredy[j]
			v2 := (v - ev) * (v - ev)
			buf[2] = buf[2] + v2
			buf[3] = buf[3] + math.Abs(v - ev);
			if ev != 0 {
				buf[4] = buf[4] + math.Abs((v - ev) / ev)
				buf[offs + 2] = buf[offs + 2] + 1
			}
		}
		buf[offs + 1] = buf[offs + 1] + 1
	}
}

/*************************************************************************
See DSErrAllocate for comments on this routine.

  -- ALGLIB --
	 Copyright 11.01.2009 by Bochkanov Sergey
*************************************************************************/
func dserrfinish(buf *[]float64) {
	offs := 5
	nout := math.Abs(int(round(buf[offs])))
	if buf[offs + 1] != 0 {
		buf[0] = buf[0] / buf[offs + 1]
		buf[1] = buf[1] / buf[offs + 1]
		buf[2] = math.Sqrt(buf[2] / (nout * buf[offs + 1]))
		buf[3] = buf[3] / (nout * buf[offs + 1])
	}
	if buf[offs + 2] != (0) {
		buf[4] = buf[4] / buf[offs + 2]
	}
}

