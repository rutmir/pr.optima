package mlpbase
import (
	"fmt"
	"math"
	"math/rand"
	"pr.optima/src/core/neural/utils"
)

const (
	mlpvnum = 7
	mlpfirstversion = 0
	nfieldwidth = 4
	hlconnfieldwidth = 5
	hlnfieldwidth = 4
	chunksize = 32

	maxrealnumber = 1E300
	minrealnumber = 1E-300
)

type Multilayerperceptron struct {
	hlnetworktype int
	hlnormtype    int
	HlLayerSizes  []int
	HlConnections []int
	HlNeurons     []int
	StructInfo    []int
	Weights       []float64
	ColumnMeans   []float64
	ColumnSigmas  []float64
	Neurons       []float64
	DfdNet        []float64
	DError        []float64
	X             []float64
	Y             []float64
	Chunks        [][]float64
	NwBuf         []float64
	IntegerBuf    []int
}

func NewMlp() *Multilayerperceptron {
	return &Multilayerperceptron{
		HlLayerSizes: []int{},
		HlConnections: []int{},
		HlNeurons: []int{},
		StructInfo: []int{},
		Weights: []float64{},
		ColumnMeans: []float64{},
		ColumnSigmas: []float64{},
		Neurons: []float64{},
		DfdNet: []float64{},
		DError: []float64{},
		X: []float64{},
		Y: []float64{},
		Chunks: [][]float64{},
		NwBuf: []float64{},
		IntegerBuf: []int{}}
}

/*************************************************************************
Creates  neural  network  with  NIn  inputs,  NOut outputs, without hidden
layers, with linear output layer. Network weights are  filled  with  small
random values.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate0(nin, nout int, network *Multilayerperceptron) error {
	layerscount := 1 + 3;
	lastproc := 0

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(-5, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, 0, 0, nout, false, true);
	return nil
}

/*************************************************************************
Same  as  MLPCreate0,  but  with  one  hidden  layer  (NHid  neurons) with
non-linear activation function. Output layer is linear.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate1(nin, nhid, nout int, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3 + 3;

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(-5, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid, 0, nout, false, true)
	return nil
}

/*************************************************************************
Same as MLPCreate0, but with two hidden layers (NHid1 and  NHid2  neurons)
with non-linear activation function. Output layer is linear.
 $ALL

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreate2(nin, nhid1, nhid2, nout int, network *Multilayerperceptron) error {
	lastproc := 0;
	layerscount := 1 + 3 + 3 + 3;

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nhid2, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(-5, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid1, nhid2, nout, false, true);
	return nil
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
func MlpCreateb0(nin, nout int, b, d float64, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3

	if d >= 0 {
		d = 1
	}else {
		d = -1
	}

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(3, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, 0, 0, nout, false, false)

	//
	// Turn on ouputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = b;
		network.ColumnSigmas[i] = d;
	}
	return nil
}

/*************************************************************************
Same as MLPCreateB0 but with non-linear hidden layer.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateb1(nin, nhid, nout int, b, d float64, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3 + 3

	if d >= 0 {
		d = 1
	}else {
		d = -1
	}

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(3, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid, 0, nout, false, false)

	//
	// Turn on ouputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = b
		network.ColumnSigmas[i] = d
	}
	return nil
}

/*************************************************************************
Same as MLPCreateB0 but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreateb2(nin, nhid1, nhid2, nout int, b, d float64, network *Multilayerperceptron) error {

	lastproc := 0
	layerscount := 1 + 3 + 3 + 3

	if d >= 0 {
		d = 1
	}else {
		d = -1
	}

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nhid2, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(3, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid1, nhid2, nout, false, false)

	//
	// Turn on ouputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = b
		network.ColumnSigmas[i] = d
	}
	return nil
}

/*************************************************************************
Creates  neural  network  with  NIn  inputs,  NOut outputs, without hidden
layers with non-linear output layer. Network weights are filled with small
random values. Activation function of the output layer takes values [A,B].

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreater0(nin, nout int, a, b float64, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, 0, 0, nout, false, false)

	//
	// Turn on outputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = 0.5 * (a + b)
		network.ColumnSigmas[i] = 0.5 * (a - b)
	}
	return nil
}

/*************************************************************************
Same as MLPCreateR0, but with non-linear hidden layer.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreater1(nin, nhid, nout int, a, b float64, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3 + 3

	//
	// Allocate arrays
	//
	lsizes := make([] int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid, 0, nout, false, false)

	//
	// Turn on outputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = 0.5 * (a + b)
		network.ColumnSigmas[i] = 0.5 * (a - b)
	}
	return nil
}

/*************************************************************************
Same as MLPCreateR0, but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpCreater2(nin, nhid1, nhid2, nout int, a, b float64, network *Multilayerperceptron) error {
	lastproc := 0
	layerscount := 1 + 3 + 3 + 3

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nhid2, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, false, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid1, nhid2, nout, false, false)

	//
	// Turn on outputs shift/scaling.
	//
	for i := nin; i <= nin + nout - 1; i++ {
		network.ColumnMeans[i] = 0.5 * (a + b)
		network.ColumnSigmas[i] = 0.5 * (a - b)
	}
	return nil
}

/*************************************************************************
Creates classifier network with NIn  inputs  and  NOut  possible  classes.
Network contains no hidden layers and linear output  layer  with  SOFTMAX-
normalization  (so  outputs  sums  up  to  1.0  and  converge to posterior
probabilities).

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreatec0(nin, nout int, network *Multilayerperceptron) error {
	lastproc := 0

	if nout < 2 {
		return fmt.Errorf("MLPCreateC0: NOut<2!")
	}
	layerscount := 1 + 2 + 1

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nout - 1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addzerolayer(&lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, true, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, 0, 0, nout, true, true)
	return nil
}

/*************************************************************************
Same as MLPCreateC0, but with one non-linear hidden layer.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreatec1(nin, nhid, nout int, network *Multilayerperceptron) error {
	lastproc := 0

	if nout < 2 {
		return fmt.Errorf("MLPCreateC1: NOut<2!")
	}
	layerscount := 1 + 3 + 2 + 1

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout - 1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addzerolayer(&lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, true, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid, 0, nout, true, true)
	return nil
}

/*************************************************************************
Same as MLPCreateC0, but with two non-linear hidden layers.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCreatec2(nin, nhid1, nhid2, nout int, network *Multilayerperceptron) error {
	lastproc := 0

	if nout < 2 {
		return fmt.Errorf("MLPCreateC2: NOut<2!")
	}
	layerscount := 1 + 3 + 3 + 2 + 1

	//
	// Allocate arrays
	//
	lsizes := make([]int, layerscount - 1 + 1)
	ltypes := make([]int, layerscount - 1 + 1)
	lconnfirst := make([]int, layerscount - 1 + 1)
	lconnlast := make([]int, layerscount - 1 + 1)

	//
	// Layers
	//
	addinputlayer(nin, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addbiasedsummatorlayer(nhid1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nhid2, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	if err := addactivationlayer(1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc); err != nil {
		return err
	}
	addbiasedsummatorlayer(nout - 1, &lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)
	addzerolayer(&lsizes, &ltypes, &lconnfirst, &lconnlast, &lastproc)

	//
	// Create
	//
	if err := mlpcreate(nin, nout, &lsizes, &ltypes, &lconnfirst, &lconnlast, layerscount, true, network); err != nil {
		return err
	}
	fillhighlevelinformation(network, nin, nhid1, nhid2, nout, true, true)
	return nil
}

/*************************************************************************
Copying of neural network

INPUT PARAMETERS:
	Network1 -   original

OUTPUT PARAMETERS:
	Network2 -   copy

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpCopy(network1 *Multilayerperceptron, network2 *Multilayerperceptron) {
	network2.hlnetworktype = network1.hlnetworktype
	network2.hlnormtype = network1.hlnormtype
	network2.HlLayerSizes = utils.CloneArrayInt(network1.HlLayerSizes)
	network2.HlConnections = utils.CloneArrayInt(network1.HlConnections)
	network2.HlNeurons = utils.CloneArrayInt(network1.HlNeurons)
	network2.StructInfo = utils.CloneArrayInt(network1.StructInfo)
	network2.Weights = utils.CloneArrayFloat64(network1.Weights)
	network2.ColumnMeans = utils.CloneArrayFloat64(network1.ColumnMeans)
	network2.ColumnSigmas = utils.CloneArrayFloat64(network1.ColumnSigmas)
	network2.Neurons = utils.CloneArrayFloat64(network1.Neurons)
	network2.DfdNet = utils.CloneArrayFloat64(network1.DfdNet)
	network2.DError = utils.CloneArrayFloat64(network1.DError)
	network2.X = utils.CloneArrayFloat64(network1.X)
	network2.Y = utils.CloneArrayFloat64(network1.Y)
	network2.Chunks = utils.CloneMatrixFloat64(network1.Chunks)
	network2.NwBuf = utils.CloneArrayFloat64(network1.NwBuf)
	network2.IntegerBuf = utils.CloneArrayInt(network1.IntegerBuf)
}

/*************************************************************************
Serialization of MultiLayerPerceptron strucure

INPUT PARAMETERS:
	Network -   original

OUTPUT PARAMETERS:
	RA      -   array of real numbers which stores network,
				array[0..RLen-1]
	RLen    -   RA lenght

  -- ALGLIB --
	 Copyright 29.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpSerializeOld(network *Multilayerperceptron, ra *[]float64, rlen *int) {
	sigmalen := 0

	//
	// Unload info
	//
	ssize := network.StructInfo[0]
	nin := network.StructInfo[1]
	nout := network.StructInfo[2]
	//	ntotal := network.structinfo[3]
	wcount := network.StructInfo[4]
	if MlpIsSoftMax(network) {
		sigmalen = nin
	}else {
		sigmalen = nin + nout
	}

	//
	//  RA format:
	//      LEN         DESRC.
	//      1           RLen
	//      1           version (MLPVNum)
	//      1           StructInfo size
	//      SSize       StructInfo
	//      WCount      Weights
	//      SigmaLen    ColumnMeans
	//      SigmaLen    ColumnSigmas
	//
	*rlen = 3 + ssize + wcount + 2 * sigmalen
	*ra = make([]float64, *rlen - 1 + 1)
	(*ra)[0] = float64(*rlen)
	(*ra)[1] = mlpvnum
	(*ra)[2] = float64(ssize)
	offs := 3
	for i := 0; i <= ssize - 1; i++ {
		(*ra)[offs + i] = float64(network.StructInfo[i])
	}
	offs = offs + ssize
	i1_ := (0) - (offs)
	for i := offs; i <= offs + wcount - 1; i++ {
		(*ra)[i] = network.Weights[i + i1_]
	}
	offs = offs + wcount
	i1_ = (0) - (offs)
	for i := offs; i <= offs + sigmalen - 1; i++ {
		(*ra)[i] = network.ColumnMeans[i + i1_]
	}
	offs = offs + sigmalen
	i1_ = (0) - (offs)
	for i := offs; i <= offs + sigmalen - 1; i++ {
		(*ra)[i] = network.ColumnSigmas[i + i1_]
	}
	offs = offs + sigmalen
}

/*************************************************************************
Unserialization of MultiLayerPerceptron strucure

INPUT PARAMETERS:
	RA      -   real array which stores network

OUTPUT PARAMETERS:
	Network -   restored network

  -- ALGLIB --
	 Copyright 29.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpUnserializeOld(ra []float64, network *Multilayerperceptron) error {
	ssize := 0
	ntotal := 0
	nin := 0
	nout := 0
	wcount := 0
	sigmalen := 0

	if !(utils.RoundInt(ra[1]) == mlpvnum) {
		return fmt.Errorf("MLPUnserialize: incorrect array!")
	}
	//
	// Unload StructInfo from IA
	//
	offs := 3
	ssize = utils.RoundInt(ra[2])
	network.StructInfo = make([]int, ssize - 1 + 1)
	for i := 0; i <= ssize - 1; i++ {
		network.StructInfo[i] = utils.RoundInt(ra[offs + i])
	}
	offs += ssize

	//
	// Unload info from StructInfo
	//
	ssize = network.StructInfo[0]
	nin = network.StructInfo[1]
	nout = network.StructInfo[2]
	ntotal = network.StructInfo[3]
	wcount = network.StructInfo[4]
	if network.StructInfo[6] == 0 {
		sigmalen = nin + nout
	}else {
		sigmalen = nin
	}

	//
	// Allocate space for other fields
	//
	network.Weights = make([]float64, wcount - 1 + 1)
	network.ColumnMeans = make([]float64, sigmalen - 1 + 1)
	network.ColumnSigmas = make([]float64, sigmalen - 1 + 1)
	network.Neurons = make([]float64, ntotal - 1 + 1)
	network.Chunks = utils.MakeMatrixFloat64(3 * ntotal + 1, chunksize - 1 + 1)
	network.NwBuf = make([]float64, utils.MaxInt(wcount, 2 * nout) - 1 + 1)
	network.DfdNet = make([]float64, ntotal - 1 + 1)
	network.X = make([]float64, nin - 1 + 1)
	network.Y = make([]float64, nout - 1 + 1)
	network.DError = make([]float64, ntotal - 1 + 1)

	//
	// Copy parameters from RA
	//
	for i := 0; i <= wcount - 1; i++ {
		network.Weights[i] = ra[i + offs]
	}
	offs += wcount
	for i := 0; i <= sigmalen - 1; i++ {
		network.ColumnMeans[i] = ra[i + offs]
	}
	offs += sigmalen
	for i := 0; i <= sigmalen - 1; i++ {
		network.ColumnSigmas[i] = ra[i + offs]
	}
	offs = offs + sigmalen
	return nil
}

/*************************************************************************
Randomization of neural network weights

  -- ALGLIB --
	 Copyright 06.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpRandomize(network *Multilayerperceptron) {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	for i := 0; i <= wcount - 1; i++ {
		network.Weights[i] = rand.Float64() - 0.5
	}
}

/*************************************************************************
Randomization of neural network weights and standartisator

  -- ALGLIB --
	 Copyright 10.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpRandomizeFull(network *Multilayerperceptron) {
	nin := 0
	nout := 0
	wcount := 0
	offs := 0
	ntype := 0

	MlpProperties(network, &nin, &nout, &wcount)
	ntotal := network.StructInfo[3]
	istart := network.StructInfo[5]

	//
	// Process network
	//
	for i := 0; i <= wcount - 1; i++ {
		network.Weights[i] = rand.Float64() - 0.5
	}
	for i := 0; i <= nin - 1; i++ {
		network.ColumnMeans[i] = 2 * rand.Float64() - 1
		network.ColumnSigmas[i] = 1.5 * rand.Float64() + 0.5
	}
	if !MlpIsSoftMax(network) {
		for i := 0; i <= nout - 1; i++ {
			offs = istart + (ntotal - nout + i) * nfieldwidth
			ntype = network.StructInfo[offs + 0]
			if ntype == 0 {
				//
				// Shifts are changed only for linear outputs neurons
				//
				network.ColumnMeans[nin + i] = 2 * rand.Float64() - 1
			}
			if ntype == 0 || ntype == 3 {
				//
				// Scales are changed only for linear or bounded outputs neurons.
				// Note that scale randomization preserves sign.
				//
				network.ColumnSigmas[nin + i] = utils.SignFloat64(network.ColumnSigmas[nin + i]) * (1.5 * rand.Float64() + 0.5)
			}
		}
	}
}

/*************************************************************************
Internal subroutine.

  -- ALGLIB --
	 Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func MlpInitPreprocessor(network *Multilayerperceptron, xy [][]float64, ssize int) {
	jmax := 0
	nin := 0
	nout := 0
	wcount := 0
	ntotal := 0
	istart := 0
	offs := 0
	ntype := 0
	means := make([]float64, 0)
	sigmas := make([]float64, 0)
	s := 0.0

	MlpProperties(network, &nin, &nout, &wcount)
	ntotal = network.StructInfo[3]
	istart = network.StructInfo[5]

	//
	// Means/Sigmas
	//
	if MlpIsSoftMax(network) {
		jmax = nin - 1
	}else {
		jmax = nin + nout - 1
	}
	means = make([]float64, jmax + 1)
	sigmas = make([]float64, jmax + 1)
	for j := 0; j <= jmax; j++ {
		means[j] = 0
		for i := 0; i <= ssize - 1; i++ {
			means[j] = means[j] + xy[i][ j]
		}
		means[j] = means[j] / float64(ssize)
		sigmas[j] = 0
		for i := 0; i <= ssize - 1; i++ {
			sigmas[j] = sigmas[j] + utils.SqrFloat64(xy[i][ j] - means[j])
		}
		sigmas[j] = math.Sqrt(sigmas[j] / float64(ssize))
	}
	//
	// Inputs
	//
	for i := 0; i <= nin - 1; i++ {
		network.ColumnMeans[i] = means[i]
		network.ColumnSigmas[i] = sigmas[i]
		if network.ColumnSigmas[i] == 0 {
			network.ColumnSigmas[i] = 1
		}
	}
	//
	// Outputs
	//
	if !MlpIsSoftMax(network) {
		for i := 0; i <= nout - 1; i++ {
			offs = istart + (ntotal - nout + i) * nfieldwidth
			ntype = network.StructInfo[offs + 0]

			//
			// Linear outputs
			//
			if ntype == 0 {
				network.ColumnMeans[nin + i] = means[nin + i]
				network.ColumnSigmas[nin + i] = sigmas[nin + i]
				if network.ColumnSigmas[nin + i] == 0 {
					network.ColumnSigmas[nin + i] = 1
				}
			}

			//
			// Bounded outputs (half-interval)
			//
			if ntype == 3 {
				s = means[nin + i] - network.ColumnMeans[nin + i]
				if s == 0 {
					s = utils.SignFloat64(network.ColumnSigmas[nin + i])
				}
				if s == 0 {
					s = 1.0
				}
				network.ColumnSigmas[nin + i] = utils.SignFloat64(network.ColumnSigmas[nin + i]) * math.Abs(s)
				if network.ColumnSigmas[nin + i] == 0 {
					network.ColumnSigmas[nin + i] = 1
				}
			}
		}
	}
}

/*************************************************************************
Returns information about initialized network: number of inputs, outputs,
weights.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpProperties(network *Multilayerperceptron, nin, nout, wcount *int) {
	*nin = network.StructInfo[1]
	*nout = network.StructInfo[2]
	*wcount = network.StructInfo[4]
}

/*************************************************************************
Tells whether network is SOFTMAX-normalized (i.e. classifier) or not.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpIsSoftMax(network *Multilayerperceptron) bool {
	return network.StructInfo[6] == 1
}

/*************************************************************************
This function returns total number of layers (including input, hidden and
output layers).

  -- ALGLIB --
	 Copyright 25.03.2011 by Bochkanov Sergey
*************************************************************************/
func MlpGetLayersCount(network *Multilayerperceptron) int {
	return len(network.HlLayerSizes)
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
func MlpGetLayerSize(network *Multilayerperceptron, k int) (int, error) {
	if !(k >= 0 && k < len(network.HlLayerSizes)) {
		return -1, fmt.Errorf("MLPGetLayerSize: incorrect layer index")
	}
	return network.HlLayerSizes[k], nil
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
func MlpGetInputScaling(network *Multilayerperceptron, i int, mean, sigma *float64) error {
	if !(i >= 0 && i < network.HlLayerSizes[0]) {
		return fmt.Errorf("MLPGetInputScaling: incorrect (nonexistent) I")
	}

	*mean = network.ColumnMeans[i]
	*sigma = network.ColumnSigmas[i]
	if *sigma == 0 {
		*sigma = 1
	}
	return nil
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
func MlpGetOutputScaling(network *Multilayerperceptron, i int, mean, sigma *float64) error {
	if !(i >= 0 && i < network.HlLayerSizes[len(network.HlLayerSizes) - 1]) {
		return fmt.Errorf("MLPGetOutputScaling: incorrect (nonexistent) I")
	}
	if network.StructInfo[6] == 1 {
		*mean = 0;
		*sigma = 1;
	}else {
		*mean = network.ColumnMeans[network.HlLayerSizes[0] + i]
		*sigma = network.ColumnSigmas[network.HlLayerSizes[0] + i]
	}
	return nil
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
func MlpGetNeuronInfo(network *Multilayerperceptron, k, i int, fkind *int, threshold *float64) error {
	*fkind = 0
	*threshold = 0
	ncnt := len(network.HlNeurons) / hlnfieldwidth
	istart := network.StructInfo[5]

	//
	// search
	//
	network.IntegerBuf[0] = k;
	network.IntegerBuf[1] = i;
	highlevelidx := recsearch(network.HlNeurons, hlnfieldwidth, 2, 0, ncnt, network.IntegerBuf)

	if !(highlevelidx >= 0) {
		return fmt.Errorf("MLPGetNeuronInfo: incorrect (nonexistent) layer or neuron index")
	}

	//
	// 1. find offset of the activation function record in the
	//
	if network.HlNeurons[highlevelidx * hlnfieldwidth + 2] >= 0 {
		activationoffset := istart + network.HlNeurons[highlevelidx * hlnfieldwidth + 2] * nfieldwidth
		*fkind = network.StructInfo[activationoffset + 0]
	}else {
		*fkind = 0
	}
	if network.HlNeurons[highlevelidx * hlnfieldwidth + 3] >= 0 {
		*threshold = network.Weights[network.HlNeurons[highlevelidx * hlnfieldwidth + 3]]
	}else {
		*threshold = 0
	}
	return nil
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
func MlpGetWeight(network *Multilayerperceptron, k0, i0, k1, i1 int) (float64, error) {
	//
	// check params
	//
	if !(k0 >= 0 && k0 < len(network.HlLayerSizes)) {
		return .0, fmt.Errorf("MLPGetWeight: incorrect (nonexistent) K0")
	}
	if !(i0 >= 0 && i0 < network.HlLayerSizes[k0]) {
		return .0, fmt.Errorf("MLPGetWeight: incorrect (nonexistent) K0")
	}
	if !(k1 >= 0 && k1 < len(network.HlLayerSizes)) {
		return .0, fmt.Errorf("MLPGetWeight: incorrect (nonexistent) K1")
	}
	if !(i1 >= 0 && i1 < network.HlLayerSizes[k1]) {
		return .0, fmt.Errorf("MLPGetWeight: incorrect (nonexistent) I1")
	}

	result := .0
	highlevelidx := 0
	ccnt := len(network.HlConnections) / hlconnfieldwidth
	//
	// search
	//
	network.IntegerBuf[0] = k0
	network.IntegerBuf[1] = i0
	network.IntegerBuf[2] = k1
	network.IntegerBuf[3] = i1
	highlevelidx = recsearch(network.HlConnections, hlconnfieldwidth, 4, 0, ccnt, network.IntegerBuf)
	if highlevelidx >= 0 {
		result = network.Weights[network.HlConnections[highlevelidx * hlconnfieldwidth + 4]]
	}else {
		result = 0
	}
	return result, nil;
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
func MlpSetInputScaling(network *Multilayerperceptron, i int, mean, sigma float64) error {
	if !(i >= 0 && i < network.HlLayerSizes[0]) {
		return fmt.Errorf("MLPSetInputScaling: incorrect (nonexistent) I")
	}
	if !(utils.IsFinite(mean)) {
		return fmt.Errorf("MLPSetInputScaling: infinite or NAN Mean")
	}
	if !(utils.IsFinite(sigma)) {
		return fmt.Errorf("MLPSetInputScaling: infinite or NAN Sigma")
	}

	if sigma == 0 {
		sigma = 1
	}
	network.ColumnMeans[i] = mean
	network.ColumnSigmas[i] = sigma
	return nil
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
func MlpSetOutputScaling(network *Multilayerperceptron, i int, mean, sigma float64) error {
	if !(i >= 0 && i < network.HlLayerSizes[len(network.HlLayerSizes) - 1]) {
		return fmt.Errorf("MLPSetOutputScaling: incorrect (nonexistent) I")
	}
	if !(utils.IsFinite(mean)) {
		return fmt.Errorf("MLPSetOutputScaling: infinite or NAN Mean")
	}
	if !(utils.IsFinite(sigma)) {
		return fmt.Errorf("MLPSetOutputScaling: infinite or NAN Sigma")
	}

	if network.StructInfo[6] == 1 {
		if !(mean == 0) {
			return fmt.Errorf("MLPSetOutputScaling: you can not set non-zero Mean term for classifier network")
		}
		if !(sigma == 1) {
			return fmt.Errorf("MLPSetOutputScaling: you can not set non-unit Sigma term for classifier network")
		}
	}else {
		if sigma == 0 {
			sigma = 1
		}
		network.ColumnMeans[network.HlLayerSizes[0] + i] = mean
		network.ColumnSigmas[network.HlLayerSizes[0] + i] = sigma
	}
	return nil
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
func MlpSetNeuronInfo(network *Multilayerperceptron, k, i, fkind int, threshold float64) error {
	if !(utils.IsFinite(threshold)) {
		return fmt.Errorf("MLPSetNeuronInfo: infinite or NAN Threshold")
	}

	//
	// convenience vars
	//
	ncnt := len(network.HlNeurons) / hlnfieldwidth
	istart := network.StructInfo[5]

	//
	// search
	//
	network.IntegerBuf[0] = k
	network.IntegerBuf[1] = i
	highlevelidx := recsearch(network.HlNeurons, hlnfieldwidth, 2, 0, ncnt, network.IntegerBuf)

	if !(highlevelidx >= 0) {
		return fmt.Errorf("MLPSetNeuronInfo: incorrect (nonexistent) layer or neuron index")
	}

	//
	// activation function
	//
	if network.HlNeurons[highlevelidx * hlnfieldwidth + 2] >= 0 {
		activationoffset := istart + network.HlNeurons[highlevelidx * hlnfieldwidth + 2] * nfieldwidth
		network.StructInfo[activationoffset + 0] = fkind
	}else {
		if !(fkind == 0) {
			return fmt.Errorf("MLPSetNeuronInfo: you try to set activation function for neuron which can not have one")
		}
	}

	//
	// Threshold
	//
	if network.HlNeurons[highlevelidx * hlnfieldwidth + 3] >= 0 {
		network.Weights[network.HlNeurons[highlevelidx * hlnfieldwidth + 3]] = threshold
	}else {
		if !(threshold == 0) {
			return fmt.Errorf("MLPSetNeuronInfo: you try to set non-zero threshold for neuron which can not have one")
		}
	}
	return nil
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
func MlpSetWeight(network *Multilayerperceptron, k0, i0, k1, i1 int, w float64) error {
	highlevelidx := 0

	ccnt := len(network.HlConnections) / hlconnfieldwidth

	//
	// check params
	//
	if !(k0 >= 0 && k0 < len(network.HlLayerSizes)) {
		return fmt.Errorf("MLPSetWeight: incorrect (nonexistent) K0")
	}
	if !(i0 >= 0 && i0 < network.HlLayerSizes[k0]) {
		return fmt.Errorf("MLPSetWeight: incorrect (nonexistent) I0")
	}
	if !(k1 >= 0 && k1 < len(network.HlLayerSizes)) {
		return fmt.Errorf("MLPSetWeight: incorrect (nonexistent) K1")
	}
	if !(i1 >= 0 && i1 < network.HlLayerSizes[k1]) {
		return fmt.Errorf("MLPSetWeight: incorrect (nonexistent) I1")
	}
	if !(utils.IsFinite(w)) {
		return fmt.Errorf("MLPSetWeight: infinite or NAN weight")
	}

	//
	// search
	//
	network.IntegerBuf[0] = k0
	network.IntegerBuf[1] = i0
	network.IntegerBuf[2] = k1
	network.IntegerBuf[3] = i1
	highlevelidx = recsearch(network.HlConnections, hlconnfieldwidth, 4, 0, ccnt, network.IntegerBuf);
	if highlevelidx >= 0 {
		network.Weights[network.HlConnections[highlevelidx * hlconnfieldwidth + 4]] = w
	}else {
		if !(w == 0) {
			return fmt.Errorf("MLPSetWeight: you try to set non-zero weight for non-existent connection")
		}
	}
	return nil
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
func MlpActivationFunction(net float64, k int, f, df, d2f *float64) {
	var net2 float64 = 0
	var arg float64 = 0
	var root float64 = 0
	var r float64 = 0

	*f = 0
	*df = 0
	*d2f = 0

	if k == 0 || k == -5 {
		*f = net
		*df = 1
		*d2f = 0
		return
	}
	if k == 1 {
		//
		// TanH activation function
		//
		if math.Abs(net) < 100.0 {
			*f = math.Tanh(net)
		}else {
			*f = utils.SignFloat64(net)
		}
		*df = 1 - utils.SqrFloat64(*f)
		*d2f = -(2 * *f * *df)
		return
	}
	if k == 3 {
		//
		// EX activation function
		//
		if net >= (0) {
			net2 = net * net
			arg = net2 + 1
			root = math.Sqrt(arg)
			*f = net + root
			r = net / root
			*df = 1 + r
			*d2f = (root - net * r) / arg
		}else {
			*f = math.Exp(net)
			*df = *f
			*d2f = *f
		}
		return
	}
	if k == 2 {
		*f = math.Exp(-utils.SqrFloat64(net))
		*df = -(2 * net * *f)
		*d2f = -(2 * (*f + *df * net))
		return
	}
	*f = 0
	*df = 0
	*d2f = 0
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
func MlpProcess(network *Multilayerperceptron, x, y *[]float64) {
	if len(*y) < network.StructInfo[2] {
		*y = make([]float64, network.StructInfo[2])
	}
	MlpInternalProcessVector(&network.StructInfo, &network.Weights, &network.ColumnMeans, &network.ColumnSigmas, &network.Neurons, &network.DfdNet, x, y)
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
func MlpProcessi(network *Multilayerperceptron, x, y *[]float64) {
	*y = make([]float64, 0)
	MlpProcess(network, x, y)
}

/*************************************************************************
Error function for neural network, internal subroutine.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpError(network *Multilayerperceptron, xy *[][]float64, ssize int) float64 {
	nin := 0
	nout := 0
	wcount := 0
	var e float64 = 0

	MlpProperties(network, &nin, &nout, &wcount)
	var result float64 = 0.0
	for i := 0; i <= ssize - 1; i++ {
		for j := 0; j <= nin - 1; j++ {
			network.X[j] = (*xy)[i][j]
		}
		MlpProcess(network, &network.X, &network.Y);
		if MlpIsSoftMax(network) {
			//
			// class labels outputs
			//
			k := utils.RoundInt((*xy)[i][nin])
			if k >= 0 && k < nout {
				network.Y[k] = network.Y[k] - 1
			}
		}else {
			//
			// real outputs
			//
			for j := 0; j <= nout - 1; j++ {
				network.Y[j] = network.Y[j] - (*xy)[i][ j + nin]
			}
		}
		e = 0.0
		for j := 0; j <= nout - 1; j++ {
			e += network.Y[j] * network.Y[j]
		}
		result = result + e / 2
	}
	return result
}

/*************************************************************************
Natural error function for neural network, internal subroutine.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpErrorN(network *Multilayerperceptron, xy *[][]float64, ssize int) float64 {
	k := 0
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	var result float64 = 0.0
	for i := 0; i <= ssize - 1; i++ {
		//
		// Process vector
		//
		for j := 0; j <= nin - 1; j++ {
			network.X[j] = (*xy)[i][j]
		}
		MlpProcess(network, &network.X, &network.Y)

		//
		// Update error function
		//
		if network.StructInfo[6] == 0 {
			//
			// Least squares error function
			//
			for j := 0; j <= nout - 1; j++ {
				network.Y[j] = network.Y[j] - (*xy)[i][ j + nin]
			}
			e := 0.0
			for j := 0; j <= nout - 1; j++ {
				e += network.Y[j] * network.Y[j]
			}
			result = result + e / 2
		}else {
			//
			// Cross-entropy error function
			//
			k = utils.RoundInt((*xy)[i][nin])
			if k >= 0 && k < nout {
				result = result + safecrossentropy(1, network.Y[k])
			}
		}
	}
	return result
}

/*************************************************************************
Classification error

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func MlpClsError(network *Multilayerperceptron, xy *[][]float64, ssize int) int {
	nin := 0
	nout := 0
	wcount := 0
	nn := 0
	ns := 0
	nmax := 0

	MlpProperties(network, &nin, &nout, &wcount)
	workx := make([]float64, nin - 1 + 1)
	worky := make([]float64, nout - 1 + 1)
	result := 0
	for i := 0; i <= ssize - 1; i++ {
		//
		// Process
		//
		for j := 0; j <= nin - 1; j++ {
			workx[j] = (*xy)[i][j]
		}
		MlpProcess(network, &workx, &worky)

		//
		// Network version of the answer
		//
		nmax = 0
		for j := 0; j <= nout - 1; j++ {
			if worky[j] > worky[nmax] {
				nmax = j
			}
		}
		nn = nmax

		//
		// Right answer
		//
		if MlpIsSoftMax(network) {
			ns = utils.RoundInt((*xy)[i][nin])
		}else {
			nmax = 0
			for j := 0; j <= nout - 1; j++ {
				if (*xy)[i][nin + j] > (*xy)[i][nin + nmax] {
					nmax = j
				}
			}
			ns = nmax
		}

		//
		// compare
		//
		if nn != ns {
			result = result + 1
		}
	}
	return result
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
func MlpRelClsError(network *Multilayerperceptron, xy *[][]float64, npoints int) float64 {
	return float64(MlpClsError(network, xy, npoints)) / float64(npoints)
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
func MlpAvgce(network *Multilayerperceptron, xy *[][]float64, npoints int) float64 {
	result := 0.0
	nin := 0
	nout := 0
	wcount := 0

	if MlpIsSoftMax(network) {
		MlpProperties(network, &nin, &nout, &wcount)
		result = MlpErrorN(network, xy, npoints) / (float64(npoints) * math.Log(2))
	}else {
		result = 0
	}
	return result;
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
func MlpRmsError(network *Multilayerperceptron, xy *[][]float64, npoints int) float64 {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	return math.Sqrt(2 * MlpError(network, xy, npoints) / float64(npoints * nout))
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
func MlpAvgError(network *Multilayerperceptron, xy *[][]float64, npoints int) float64 {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	result := 0.0
	for i := 0; i <= npoints - 1; i++ {
		for j := 0; j <= nin - 1; j++ {
			network.X[j] = (*xy)[i][j]
		}
		MlpProcess(network, &network.X, &network.Y)
		if MlpIsSoftMax(network) {
			//
			// class labels
			//
			k := utils.RoundInt((*xy)[i][nin])
			for j := 0; j <= nout - 1; j++ {
				if j == k {
					result = result + math.Abs(1 - network.Y[j])
				}else {
					result = result + math.Abs(network.Y[j])
				}
			}
		}else {
			//
			// real outputs
			//
			for j := 0; j <= nout - 1; j++ {
				result = result + math.Abs((*xy)[i][nin + j] - network.Y[j])
			}
		}
	}
	result = result / float64(npoints * nout)
	return result
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
func MlpAvgRelError(network *Multilayerperceptron, xy *[][]float64, npoints int) float64 {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	result := 0.0
	k := 0
	for i := 0; i <= npoints - 1; i++ {
		for j := 0; j <= nin - 1; j++ {
			network.X[j] = (*xy)[i][j]
		}
		MlpProcess(network, &network.X, &network.Y)
		if MlpIsSoftMax(network) {
			//
			// class labels
			//
			lk := utils.RoundInt((*xy)[i][nin])
			for j := 0; j <= nout - 1; j++ {
				if j == lk {
					result = result + math.Abs(1 - network.Y[j])
					k = k + 1
				}
			}
		}else {
			//
			// real outputs
			//
			for j := 0; j <= nout - 1; j++ {
				if (*xy)[i][nin + j] != 0 {
					result = result + math.Abs((*xy)[i][nin + j] - network.Y[j]) / math.Abs((*xy)[i][nin + j])
					k += 1
				}
			}
		}
	}
	if k != 0 {
		result /= float64(k)
	}
	return result
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
func MlpGrad(network *Multilayerperceptron, x, desiredy *[]float64, e *float64, grad *[]float64) error {
	//
	// Alloc
	//
	if len(*grad) < network.StructInfo[4] {
		*grad = make([]float64, network.StructInfo[4])
	}

	//
	// Prepare dError/dOut, internal structures
	//
	MlpProcess(network, x, &network.Y)
	nout := network.StructInfo[2]
	ntotal := network.StructInfo[3]
	*e = 0
	for i := 0; i <= ntotal - 1; i++ {
		network.DError[i] = 0
	}
	for i := 0; i <= nout - 1; i++ {
		network.DError[ntotal - nout + i] = network.Y[i] - (*desiredy)[i]
		*e += utils.SqrFloat64(network.Y[i] - (*desiredy)[i]) / 2
	}

	//
	// gradient
	//
	return mlpinternalcalculategradient(network, &network.Neurons, &network.Weights, &network.DError, grad, false)
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
func MlpGradn(network *Multilayerperceptron, x, desiredy *[]float64, e *float64, grad *[]float64) error {
	s := 0.0
	nout := 0
	ntotal := 0

	*e = 0

	//
	// Alloc
	//
	if len(*grad) < network.StructInfo[4] {
		*grad = make([]float64, network.StructInfo[4])
	}

	//
	// Prepare dError/dOut, internal structures
	//
	MlpProcess(network, x, &network.Y)
	nout = network.StructInfo[2]
	ntotal = network.StructInfo[3]
	for i := 0; i <= ntotal - 1; i++ {
		network.DError[i] = 0
	}
	*e = 0
	if network.StructInfo[6] == 0 {
		//
		// Regression network, least squares
		//
		for i := 0; i <= nout - 1; i++ {
			network.DError[ntotal - nout + i] = network.Y[i] - (*desiredy)[i]
			*e += utils.SqrFloat64(network.Y[i] - (*desiredy)[i]) / 2
		}
	}else {
		//
		// Classification network, cross-entropy
		//
		s = 0
		for i := 0; i <= nout - 1; i++ {
			s += (*desiredy)[i]
		}
		for i := 0; i <= nout - 1; i++ {
			network.DError[ntotal - nout + i] = s * network.Y[i] - (*desiredy)[i]
			*e += safecrossentropy((*desiredy)[i], network.Y[i])
		}
	}

	//
	// gradient
	//
	return mlpinternalcalculategradient(network, &network.Neurons, &network.Weights, &network.DError, grad, true)
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
func MlpGradBatch(network *Multilayerperceptron, xy [][]float64, ssize int, e *float64, grad *[]float64) error {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	for i := 0; i <= wcount - 1; i++ {
		(*grad)[i] = 0
	}
	*e = 0.0

	for i := 0; i <= ssize - 1; i += chunksize {
		if err := mlpchunkedgradient(network, xy, i, utils.MinInt(ssize, i + chunksize) - i, e, grad, false); err != nil {
			return err
		}
	}
	return nil
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
func MlpGradNBatch(network *Multilayerperceptron, xy [][]float64, ssize int, e *float64, grad *[]float64) error {
	nin := 0
	nout := 0
	wcount := 0

	MlpProperties(network, &nin, &nout, &wcount)
	for i := 0; i <= wcount - 1; i++ {
		(*grad)[i] = 0
	}
	*e = 0
	i := 0
	for i <= ssize - 1 {
		if err := mlpchunkedgradient(network, xy, i, utils.MinInt(ssize, i + chunksize) - i, e, grad, true); err != nil {
			return err
		}
		i += chunksize
	}
	return nil
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
func MlpHessianNBatch(network *Multilayerperceptron, xy *[][]float64, ssize int, e *float64, grad *[]float64, h *[][]float64) error {
	*e = 0
	return mlphessianbatchinternal(network, xy, ssize, true, e, grad, h)
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
func MlpHessianBatch(network *Multilayerperceptron, xy *[][]float64, ssize int, e *float64, grad *[]float64, h *[][]float64) error {
	*e = 0
	return mlphessianbatchinternal(network, xy, ssize, false, e, grad, h)
}

/*************************************************************************
Internal subroutine, shouldn't be called by user.
*************************************************************************/
func MlpInternalProcessVector(structinfo *[]int, weights, columnmeans, columnsigmas, neurons, dfdnet, x, y *[]float64) error {
	//
	// Read network geometry
	//
	nin := (*structinfo)[1]
	nout := (*structinfo)[2]
	ntotal := (*structinfo)[3]
	istart := (*structinfo)[5]

	//
	// Inputs standartisation and putting in the network
	//
	for i := 0; i <= nin - 1; i++ {
		if (*columnsigmas)[i] != 0 {
			(*neurons)[i] = ((*x)[i] - (*columnmeans)[i]) / (*columnsigmas)[i]
		}else {
			(*neurons)[i] = (*x)[i] - (*columnmeans)[i]
		}
	}

	//
	// Process network
	//
	for i := 0; i <= ntotal - 1; i++ {
		offs := istart + i * nfieldwidth
		if (*structinfo)[offs + 0] > 0 || (*structinfo)[offs + 0] == -5 {
			//
			// Activation function
			//
			var f float64 = 0
			var df float64 = 0
			var d2f float64 = 0
			MlpActivationFunction((*neurons)[(*structinfo)[offs + 2]], (*structinfo)[offs + 0], &f, &df, &d2f)
			(*neurons)[i] = f
			(*dfdnet)[i] = df
			continue
		}
		if (*structinfo)[offs + 0] == 0 {
			//
			// Adaptive summator
			//
			n1 := (*structinfo)[offs + 2]
			// n2 := n1 + structinfo[offs + 1] - 1
			w1 := (*structinfo)[offs + 3]
			w2 := w1 + (*structinfo)[offs + 1] - 1
			l := n1 - w1
			net := 0.0
			for j := w1; j <= w2; j++ {
				net += (*weights)[j] * (*neurons)[j + l]
			}
			(*neurons)[i] = net
			(*dfdnet)[i] = 1.0
			continue
		}
		if (*structinfo)[offs + 0] < 0 {
			perr := true
			if (*structinfo)[offs + 0] == -2 {
				//
				// input neuron, left unchanged
				//
				perr = false
			}
			if (*structinfo)[offs + 0] == -3 {
				//
				// "-1" neuron
				//
				(*neurons)[i] = -1
				perr = false
			}
			if (*structinfo)[offs + 0] == -4 {
				//
				// "0" neuron
				//
				(*neurons)[i] = 0
				perr = false
			}
			if perr {
				return fmt.Errorf("MLPInternalProcessVector: internal error - unknown neuron type!")
			}
			continue
		}
	}

	//
	// Extract result
	//
	l := (ntotal - nout) - (0)
	for j := 0; j <= nout - 1; j++ {
		(*y)[j] = (*neurons)[j + l]
	}

	//
	// Softmax post-processing or standardisation if needed
	//
	if !((*structinfo)[6] == 0 || (*structinfo)[6] == 1) {
		return fmt.Errorf("MLPInternalProcessVector: unknown normalization type!")
	}
	if (*structinfo)[6] == 1 {
		//
		// Softmax
		//
		mx := (*y)[0]
		for i := 1; i <= nout - 1; i++ {
			mx = math.Max(mx, (*y)[i])
		}
		net := 0.0
		for i := 0; i <= nout - 1; i++ {
			(*y)[i] = math.Exp((*y)[i] - mx)
			net += (*y)[i]
		}
		for i := 0; i <= nout - 1; i++ {
			(*y)[i] /= net
		}
	}else {
		//
		// Standardisation
		//
		for i := 0; i <= nout - 1; i++ {
			(*y)[i] = (*y)[i] * (*columnsigmas)[nin + i] + (*columnmeans)[nin + i]
		}
	}
	return nil
}

/*************************************************************************
Internal subroutine: adding new input layer to network
*************************************************************************/
func addinputlayer(ncount int, lsizes, ltypes, lconnfirst, lconnlast *[]int, lastproc *int) {
	(*lsizes)[0] = ncount
	(*ltypes)[0] = -2
	(*lconnfirst)[0] = 0
	(*lconnlast)[0] = 0
	*lastproc = 0
}

/*************************************************************************
Internal subroutine: adding new summator layer to network
*************************************************************************/
func addbiasedsummatorlayer(ncount int, lsizes, ltypes, lconnfirst, lconnlast *[]int, lastproc *int) {
	(*lsizes)[*lastproc + 1] = 1
	(*ltypes)[*lastproc + 1] = -3
	(*lconnfirst)[*lastproc + 1] = 0
	(*lconnlast)[*lastproc + 1] = 0
	(*lsizes)[*lastproc + 2] = ncount
	(*ltypes)[*lastproc + 2] = 0
	(*lconnfirst)[*lastproc + 2] = *lastproc
	(*lconnlast)[*lastproc + 2] = *lastproc + 1
	*lastproc = *lastproc + 2
}

/*************************************************************************
Internal subroutine: adding new summator layer to network
*************************************************************************/
func addactivationlayer(functype int, lsizes, ltypes, lconnfirst, lconnlast *[]int, lastproc *int) error {
	if functype <= 0 && functype != -5 {
		return fmt.Errorf("AddActivationLayer: incorrect function type %d", functype)
	}
	(*lsizes)[*lastproc + 1] = (*lsizes)[*lastproc]
	(*ltypes)[*lastproc + 1] = functype
	(*lconnfirst)[*lastproc + 1] = *lastproc
	(*lconnlast)[*lastproc + 1] = *lastproc
	*lastproc = *lastproc + 1
	return nil
}

/*************************************************************************
Internal subroutine: adding new zero layer to network
*************************************************************************/
func addzerolayer(lsizes, ltypes, lconnfirst, lconnlast *[]int, lastproc *int) {
	(*lsizes)[*lastproc + 1] = 1
	(*ltypes)[*lastproc + 1] = -4
	(*lconnfirst)[*lastproc + 1] = 0
	(*lconnlast)[*lastproc + 1] = 0
	*lastproc += 1
}

/*************************************************************************
Internal subroutine.

  -- ALGLIB --
	 Copyright 04.11.2007 by Bochkanov Sergey
*************************************************************************/
func mlpcreate(nin, nout int, lsizes, ltypes, lconnfirst, lconnlast *[]int, layerscount int, isclsnet bool, network *Multilayerperceptron) error {
	ssize := 0
	offs := 0
	nprocessed := 0
	wallocated := 0

	//
	// Check
	//
	if !(layerscount > 0) {
		return fmt.Errorf("MLPCreate: wrong parameters! (layerscount %d)", layerscount)
	}
	if !((*ltypes)[0] == -2) {
		return fmt.Errorf("MLPCreate: wrong LTypes[0] (must be -2)!")
	}

	for i := 0; i <= layerscount - 1; i++ {
		if (*lsizes)[i] <= 0 {
			return fmt.Errorf("MLPCreate: wrong LSizes!")
		}
		if (*lconnfirst)[i] < 0 || ((*lconnfirst)[i] >= i && i != 0) {
			return fmt.Errorf("MLPCreate: wrong LConnFirst!")
		}
		if (*lconnlast)[i] < (*lconnfirst)[i] || ((*lconnlast)[i] >= i && i != 0) {
			return fmt.Errorf("MLPCreate: wrong LConnLast!")
		}
	}

	//
	// Build network geometry
	//
	lnfirst := make([]int, layerscount - 1 + 1)
	lnsyn := make([]int, layerscount - 1 + 1)
	ntotal := 0
	wcount := 0

	for i := 0; i <= layerscount - 1; i++ {
		//
		// Analyze connections.
		// This code must throw an assertion in case of unknown LTypes[I]
		//
		lnsyn[i] = -1;
		if (*ltypes)[i] >= 0 || (*ltypes)[i] == -5 {
			lnsyn[i] = 0
			for j := (*lconnfirst)[i]; j <= (*lconnlast)[i]; j++ {
				lnsyn[i] = lnsyn[i] + (*lsizes)[j]
			}
		}else {
			if ((*ltypes)[i] == -2 || (*ltypes)[i] == -3) || (*ltypes)[i] == -4 {
				lnsyn[i] = 0
			}
		}
		if lnsyn[i] < 0 {
			return fmt.Errorf("MLPCreate: internal error #0!")
		}

		//
		// Other info
		//
		lnfirst[i] = ntotal
		ntotal = ntotal + (*lsizes)[i]
		if (*ltypes)[i] == 0 {
			wcount = wcount + lnsyn[i] * (*lsizes)[i]
		}
	}
	ssize = 7 + ntotal * nfieldwidth

	//
	// Allocate
	//
	network.StructInfo = make([]int, ssize - 1 + 1)
	network.Weights = make([]float64, wcount - 1 + 1)
	if isclsnet {
		network.ColumnMeans = make([]float64, nin - 1 + 1)
		network.ColumnSigmas = make([]float64, nin - 1 + 1)
	}else {
		network.ColumnMeans = make([]float64, nin + nout - 1 + 1)
		network.ColumnSigmas = make([]float64, nin + nout - 1 + 1)
	}
	network.Neurons = make([]float64, ntotal - 1 + 1)
	network.Chunks = utils.MakeMatrixFloat64(3 * ntotal + 1, chunksize - 1 + 1)
	network.NwBuf = make([]float64, utils.MaxInt(wcount, 2 * nout) - 1 + 1)
	network.IntegerBuf = make([]int, 3 + 1)
	network.DfdNet = make([]float64, ntotal - 1 + 1)
	network.X = make([]float64, nin - 1 + 1)
	network.Y = make([]float64, nout - 1 + 1)
	network.DError = make([]float64, ntotal - 1 + 1)

	//
	// Fill structure: global info
	//
	network.StructInfo[0] = ssize
	network.StructInfo[1] = nin
	network.StructInfo[2] = nout
	network.StructInfo[3] = ntotal
	network.StructInfo[4] = wcount
	network.StructInfo[5] = 7
	if isclsnet {
		network.StructInfo[6] = 1
	}else {
		network.StructInfo[6] = 0
	}

	//
	// Fill structure: neuron connections
	//
	nprocessed = 0
	wallocated = 0
	for i := 0; i <= layerscount - 1; i++ {
		for j := 0; j <= (*lsizes)[i] - 1; j++ {
			offs = network.StructInfo[5] + nprocessed * nfieldwidth
			network.StructInfo[offs + 0] = (*ltypes)[i]
			if (*ltypes)[i] == 0 {
				//
				// Adaptive summator:
				// * connections with weights to previous neurons
				//
				network.StructInfo[offs + 1] = lnsyn[i]
				network.StructInfo[offs + 2] = lnfirst[(*lconnfirst)[i]]
				network.StructInfo[offs + 3] = wallocated
				wallocated = wallocated + lnsyn[i]
				nprocessed = nprocessed + 1
			}
			if (*ltypes)[i] > 0 || (*ltypes)[i] == -5 {
				//
				// Activation layer:
				// * each neuron connected to one (only one) of previous neurons.
				// * no weights
				//
				network.StructInfo[offs + 1] = 1
				network.StructInfo[offs + 2] = lnfirst[(*lconnfirst)[i]] + j
				network.StructInfo[offs + 3] = -1
				nprocessed = nprocessed + 1
			}
			if ((*ltypes)[i] == -2 || (*ltypes)[i] == -3) || (*ltypes)[i] == -4 {
				nprocessed += 1
			}
		}
	}
	if wallocated != wcount {
		return fmt.Errorf("MLPCreate: internal error #1!")
	}
	if nprocessed != ntotal {
		return fmt.Errorf("MLPCreate: internal error #2!")
	}

	//
	// Fill weights by small random values
	// Initialize means and sigmas
	//
	for i := 0; i <= wcount - 1; i++ {
		network.Weights[i] = rand.Float64() - 0.5
	}
	for i := 0; i <= nin - 1; i++ {
		network.ColumnMeans[i] = 0
		network.ColumnSigmas[i] = 1
	}
	if !isclsnet {
		for i := 0; i <= nout - 1; i++ {
			network.ColumnMeans[nin + i] = 0
			network.ColumnSigmas[nin + i] = 1
		}
	}

	return nil
}

/*************************************************************************
This function fills high level information about network created using
internal MLPCreate() function.

This function does NOT examine StructInfo for low level information, it
just expects that network has following structure:

    input neuron            \
    ...                      | input layer
    input neuron            /

    "-1" neuron             \
    biased summator          |
    ...                      |
    biased summator          | hidden layer(s), if there are exists any
    activation function      |
    ...                      |
    activation function     /

    "-1" neuron            \
    biased summator         | output layer:
    ...                     |
    biased summator         | * we have NOut summators/activators for regression networks
    activation function     | * we have only NOut-1 summators and no activators for classifiers
    ...                     | * we have "0" neuron only when we have classifier
    activation function     |
    "0" neuron              /

  -- ALGLIB --
     Copyright 30.03.2008 by Bochkanov Sergey
*************************************************************************/
func fillhighlevelinformation(network *Multilayerperceptron, nin, nhid1, nhid2, nout int, iscls, islinearout bool) error {
	if !((iscls && islinearout) || !iscls) {
		return fmt.Errorf("FillHighLevelInformation: internal error")
	}

	//
	// Preparations common to all types of networks
	//
	idxweights := 0
	idxstruct := 0
	idxneuro := 0
	idxconn := 0
	network.hlnetworktype = 0

	//
	// network without hidden layers
	//
	if nhid1 == 0 {
		network.HlLayerSizes = make([]int, 2)
		network.HlLayerSizes[0] = nin
		network.HlLayerSizes[1] = nout
		if !iscls {
			network.HlConnections = make([]int, hlconnfieldwidth * nin * nout)
			network.HlNeurons = make([]int, hlnfieldwidth * (nin + nout))
			network.hlnormtype = 0
		}else {
			network.HlConnections = make([]int, hlconnfieldwidth * nin * (nout - 1))
			network.HlNeurons = make([]int, hlnfieldwidth * (nin + nout))
			network.hlnormtype = 1
		}
		hladdinputlayer(network, &idxconn, &idxneuro, &idxstruct, nin)
		return hladdoutputlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 1, nin, nout, iscls, islinearout)
	}

	//
	// network with one hidden layers
	//
	if nhid2 == 0 {
		network.HlLayerSizes = make([]int, 3)
		network.HlLayerSizes[0] = nin
		network.HlLayerSizes[1] = nhid1
		network.HlLayerSizes[2] = nout
		if !iscls {
			network.HlConnections = make([]int, hlconnfieldwidth * (nin * nhid1 + nhid1 * nout))
			network.HlNeurons = make([]int, hlnfieldwidth * (nin + nhid1 + nout))
			network.hlnormtype = 0
		}else {
			network.HlConnections = make([]int, hlconnfieldwidth * (nin * nhid1 + nhid1 * (nout - 1)))
			network.HlNeurons = make([]int, hlnfieldwidth * (nin + nhid1 + nout))
			network.hlnormtype = 1
		}
		hladdinputlayer(network, &idxconn, &idxneuro, &idxstruct, nin)
		hladdhiddenlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 1, nin, nhid1)
		return hladdoutputlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 2, nhid1, nout, iscls, islinearout)
	}

	//
	// Two hidden layers
	//
	network.HlLayerSizes = make([]int, 4)
	network.HlLayerSizes[0] = nin
	network.HlLayerSizes[1] = nhid1
	network.HlLayerSizes[2] = nhid2
	network.HlLayerSizes[3] = nout
	if !iscls {
		network.HlConnections = make([]int, hlconnfieldwidth * (nin * nhid1 + nhid1 * nhid2 + nhid2 * nout))
		network.HlNeurons = make([]int, hlnfieldwidth * (nin + nhid1 + nhid2 + nout))
		network.hlnormtype = 0
	}else {
		network.HlConnections = make([]int, hlconnfieldwidth * (nin * nhid1 + nhid1 * nhid2 + nhid2 * (nout - 1)))
		network.HlNeurons = make([]int, hlnfieldwidth * (nin + nhid1 + nhid2 + nout))
		network.hlnormtype = 1
	}
	hladdinputlayer(network, &idxconn, &idxneuro, &idxstruct, nin)
	hladdhiddenlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 1, nin, nhid1)
	hladdhiddenlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 2, nhid1, nhid2)
	return hladdoutputlayer(network, &idxconn, &idxneuro, &idxstruct, &idxweights, 3, nhid2, nout, iscls, islinearout)
}

/*************************************************************************
This routine adds input layer to the high-level description of the network.

It modifies Network.HLConnections and Network.HLNeurons  and  assumes that
these  arrays  have  enough  place  to  store  data.  It accepts following
parameters:
    Network     -   network
    ConnIdx     -   index of the first free entry in the HLConnections
    NeuroIdx    -   index of the first free entry in the HLNeurons
    StructInfoIdx-  index of the first entry in the low level description
                    of the current layer (in the StructInfo array)
    NIn         -   number of inputs

It modified Network and indices.
*************************************************************************/
func hladdinputlayer(network *Multilayerperceptron, connidx, neuroidx, structinfoidx *int, nin int) {
	offs := hlnfieldwidth * *neuroidx

	for i := 0; i <= nin - 1; i++ {
		network.HlNeurons[offs + 0] = 0
		network.HlNeurons[offs + 1] = i
		network.HlNeurons[offs + 2] = -1
		network.HlNeurons[offs + 3] = -1
		offs += hlnfieldwidth
	}
	*neuroidx += nin
	*structinfoidx += nin
}

/*************************************************************************
This routine adds output layer to the high-level description of
the network.

It modifies Network.HLConnections and Network.HLNeurons  and  assumes that
these  arrays  have  enough  place  to  store  data.  It accepts following
parameters:
    Network     -   network
    ConnIdx     -   index of the first free entry in the HLConnections
    NeuroIdx    -   index of the first free entry in the HLNeurons
    StructInfoIdx-  index of the first entry in the low level description
                    of the current layer (in the StructInfo array)
    WeightsIdx  -   index of the first entry in the Weights array which
                    corresponds to the current layer
    K           -   current layer index
    NPrev       -   number of neurons in the previous layer
    NOut        -   number of outputs
    IsCls       -   is it classifier network?
    IsLinear    -   is it network with linear output?

It modified Network and ConnIdx/NeuroIdx/StructInfoIdx/WeightsIdx.
*************************************************************************/
func hladdoutputlayer(network *Multilayerperceptron, connidx, neuroidx, structinfoidx, weightsidx *int, k, nprev, nout int, iscls, islinearout bool) error {
	i := 0
	neurooffs := 0
	connoffs := 0

	if !((iscls && islinearout) || !iscls) {
		return fmt.Errorf("HLAddOutputLayer: internal error")
	}

	neurooffs = hlnfieldwidth * *neuroidx
	connoffs = hlconnfieldwidth * *connidx
	if !iscls {

		//
		// Regression network
		//
		for i = 0; i <= nout - 1; i++ {
			network.HlNeurons[neurooffs + 0] = k
			network.HlNeurons[neurooffs + 1] = i
			network.HlNeurons[neurooffs + 2] = *structinfoidx + 1 + nout + i
			network.HlNeurons[neurooffs + 3] = *weightsidx + nprev + (nprev + 1) * i
			neurooffs += hlnfieldwidth
		}
		for i := 0; i <= nprev - 1; i++ {
			for j := 0; j <= nout - 1; j++ {
				network.HlConnections[connoffs + 0] = k - 1
				network.HlConnections[connoffs + 1] = i
				network.HlConnections[connoffs + 2] = k
				network.HlConnections[connoffs + 3] = j
				network.HlConnections[connoffs + 4] = *weightsidx + i + j * (nprev + 1)
				connoffs += hlconnfieldwidth
			}
		}
		*connidx += (nprev * nout)
		*neuroidx += nout
		*structinfoidx = *structinfoidx + 2 * nout + 1
		*weightsidx = *weightsidx + nout * (nprev + 1)
	}else {
		//
		// Classification network
		//
		for i = 0; i <= nout - 2; i++ {
			network.HlNeurons[neurooffs + 0] = k
			network.HlNeurons[neurooffs + 1] = i
			network.HlNeurons[neurooffs + 2] = -1
			network.HlNeurons[neurooffs + 3] = *weightsidx + nprev + (nprev + 1) * i
			neurooffs = neurooffs + hlnfieldwidth
		}
		network.HlNeurons[neurooffs + 0] = k
		network.HlNeurons[neurooffs + 1] = i
		network.HlNeurons[neurooffs + 2] = -1
		network.HlNeurons[neurooffs + 3] = -1
		for i = 0; i <= nprev - 1; i++ {
			for j := 0; j <= nout - 2; j++ {
				network.HlConnections[connoffs + 0] = k - 1
				network.HlConnections[connoffs + 1] = i
				network.HlConnections[connoffs + 2] = k
				network.HlConnections[connoffs + 3] = j
				network.HlConnections[connoffs + 4] = *weightsidx + i + j * (nprev + 1)
				connoffs += hlconnfieldwidth
			}
		}
		*connidx = *connidx + nprev * (nout - 1)
		*neuroidx += nout
		*structinfoidx = *structinfoidx + nout + 2
		*weightsidx = *weightsidx + (nout - 1) * (nprev + 1)
	}
	return nil
}

/*************************************************************************
This routine adds hidden layer to the high-level description of
the network.

It modifies Network.HLConnections and Network.HLNeurons  and  assumes that
these  arrays  have  enough  place  to  store  data.  It accepts following
parameters:
    Network     -   network
    ConnIdx     -   index of the first free entry in the HLConnections
    NeuroIdx    -   index of the first free entry in the HLNeurons
    StructInfoIdx-  index of the first entry in the low level description
                    of the current layer (in the StructInfo array)
    WeightsIdx  -   index of the first entry in the Weights array which
                    corresponds to the current layer
    K           -   current layer index
    NPrev       -   number of neurons in the previous layer
    NCur        -   number of neurons in the current layer

It modified Network and ConnIdx/NeuroIdx/StructInfoIdx/WeightsIdx.
*************************************************************************/
func hladdhiddenlayer(network *Multilayerperceptron, connidx, neuroidx, structinfoidx, weightsidx *int, k, nprev, ncur int) {
	neurooffs := hlnfieldwidth * *neuroidx
	connoffs := hlconnfieldwidth * *connidx

	for i := 0; i <= ncur - 1; i++ {
		network.HlNeurons[neurooffs + 0] = k
		network.HlNeurons[neurooffs + 1] = i
		network.HlNeurons[neurooffs + 2] = *structinfoidx + 1 + ncur + i
		network.HlNeurons[neurooffs + 3] = *weightsidx + nprev + (nprev + 1) * i
		neurooffs = neurooffs + hlnfieldwidth
	}
	for i := 0; i <= nprev - 1; i++ {
		for j := 0; j <= ncur - 1; j++ {
			network.HlConnections[connoffs + 0] = k - 1
			network.HlConnections[connoffs + 1] = i
			network.HlConnections[connoffs + 2] = k
			network.HlConnections[connoffs + 3] = j
			network.HlConnections[connoffs + 4] = *weightsidx + i + j * (nprev + 1)
			connoffs = connoffs + hlconnfieldwidth
		}
	}
	*connidx = *connidx + nprev * ncur
	*neuroidx += ncur
	*structinfoidx = *structinfoidx + 2 * ncur + 1
	*weightsidx = *weightsidx + ncur * (nprev + 1)
}

/*************************************************************************
        This function searches integer array. Elements in this array are actually
        records, each NRec elements wide. Each record has unique header - NHeader
        integer values, which identify it. Records are lexicographically sorted by
        header.

        Records are identified by their index, not offset (offset = NRec*index).

        This function searches A (records with indices [I0,I1)) for a record with
        header B. It returns index of this record (not offset!), or -1 on failure.

          -- ALGLIB --
             Copyright 28.03.2011 by Bochkanov Sergey
        *************************************************************************/
func recsearch(a []int, nrec, nheader, i0, i1 int, b []int) int {
	mididx := 0
	cflag := 0
	offs := 0

	for {
		if i0 >= i1 {
			break
		}
		mididx = (i0 + i1) / 2
		offs = nrec * mididx
		cflag = 0
		for k := 0; k <= nheader - 1; k++ {
			if a[offs + k] < b[k] {
				cflag = -1
				break
			}
			if a[offs + k] > b[k] {
				cflag = 1
				break
			}
		}
		if cflag == 0 {
			return mididx
		}
		if cflag < 0 {
			i0 = mididx + 1
		}else {
			i1 = mididx
		}
	}
	return -1
}

/*************************************************************************
Returns T*Ln(T/Z), guarded against overflow/underflow.
Internal subroutine.
*************************************************************************/
func safecrossentropy(t, z float64) float64 {
	result := 0.0
	r := 0.0

	if t == 0 {
		result = 0.0
	}else {
		if math.Abs(z) > 1 {
			//
			// Shouldn't be the case with softmax,
			// but we just want to be sure.
			//
			if t / z == 0 {
				r = minrealnumber
			} else {
				r = t / z
			}
		}else {
			//
			// Normal case
			//
			if z == 0 || math.Abs(t) >= maxrealnumber * math.Abs(z) {
				r = maxrealnumber
			}else {
				r = t / z}
		}
		result = t * math.Log(r)
	}
	return result
}

/*************************************************************************
Internal subroutine

Network must be processed by MLPProcess on X
*************************************************************************/
func mlpinternalcalculategradient(network *Multilayerperceptron, neurons, weights, derror, grad *[]float64, naturalerrorfunc bool) error {
	n1 := 0
	n2 := 0
	w1 := 0
	w2 := 0
	ntotal := 0
	istart := 0
	nin := 0
	nout := 0
	offs := 0
	dedf := 0.0
	dfdnet := 0.0
	v := 0.0
	fown := 0.0
	deown := 0.0
	net := 0.0
	mx := 0.0
	i1_ := 0

	//
	// Read network geometry
	//
	nin = network.StructInfo[1]
	nout = network.StructInfo[2]
	ntotal = network.StructInfo[3]
	istart = network.StructInfo[5]

	//
	// Pre-processing of dError/dOut:
	// from dError/dOut(normalized) to dError/dOut(non-normalized)
	//
	if !(network.StructInfo[6] == 0 || network.StructInfo[6] == 1) {
		return fmt.Errorf("MLPInternalCalculateGradient: unknown normalization type!")
	}
	if network.StructInfo[6] == 1 {
		//
		// Softmax
		//
		if !naturalerrorfunc {
			mx = network.Neurons[ntotal - nout]
			for i := 0; i <= nout - 1; i++ {
				mx = math.Max(mx, network.Neurons[ntotal - nout + i])
			}
			net = 0
			for i := 0; i <= nout - 1; i++ {
				network.NwBuf[i] = math.Exp(network.Neurons[ntotal - nout + i] - mx)
				net = net + network.NwBuf[i]
			}
			i1_ = (0) - (ntotal - nout)
			v = 0.0
			for i := ntotal - nout; i <= ntotal - 1; i++ {
				v += network.DError[i] * network.NwBuf[i + i1_]
			}
			for i := 0; i <= nout - 1; i++ {
				fown = network.NwBuf[i]
				deown = network.DError[ntotal - nout + i]
				network.NwBuf[nout + i] = (-v + deown * fown + deown * (net - fown)) * fown / utils.SqrFloat64(net)
			}
			for i := 0; i <= nout - 1; i++ {
				network.DError[ntotal - nout + i] = network.NwBuf[nout + i]
			}
		}
	}else {
		//
		// Un-standardisation
		//
		for i := 0; i <= nout - 1; i++ {
			network.DError[ntotal - nout + i] = network.DError[ntotal - nout + i] * network.ColumnSigmas[nin + i]
		}
	}

	//
	// Backpropagation
	//
	for i := ntotal - 1; i >= 0; i-- {
		//
		// Extract info
		//
		offs = istart + i * nfieldwidth
		if network.StructInfo[offs + 0] > 0 || network.StructInfo[offs + 0] == -5 {
			//
			// Activation function
			//
			dedf = network.DError[i]
			dfdnet = network.DfdNet[i]
			(*derror)[network.StructInfo[offs + 2]] = (*derror)[network.StructInfo[offs + 2]] + dedf * dfdnet
			continue
		}
		if network.StructInfo[offs + 0] == 0 {
			//
			// Adaptive summator
			//
			n1 = network.StructInfo[offs + 2]
			n2 = n1 + network.StructInfo[offs + 1] - 1
			w1 = network.StructInfo[offs + 3]
			w2 = w1 + network.StructInfo[offs + 1] - 1
			dedf = network.DError[i]
			dfdnet = 1.0
			v = dedf * dfdnet
			i1_ = (n1) - (w1)
			for j := w1; j <= w2; j++ {
				(*grad)[j] = v * (*neurons)[j + i1_]
			}
			i1_ = (w1) - (n1)
			for j := n1; j <= n2; j++ {
				(*derror)[j] = (*derror)[j] + v * (*weights)[j + i1_]
			}
			continue
		}
		if network.StructInfo[offs + 0] < 0 {
			bflag := false
			if (network.StructInfo[offs + 0] == -2 || network.StructInfo[offs + 0] == -3) || network.StructInfo[offs + 0] == -4 {
				//
				// Special neuron type, no back-propagation required
				//
				bflag = true
			}
			if !(bflag) {
				return fmt.Errorf("MLPInternalCalculateGradient: unknown neuron type!")
			}
			continue
		}
	}
	return nil
}

/*************************************************************************
Internal subroutine, chunked gradient
*************************************************************************/

func mlpchunkedgradient(network *Multilayerperceptron, xy [][]float64, cstart, csize int, e *float64, grad *[]float64, naturalerrorfunc bool) error {
	i := 0
	j := 0
	k := 0
	kl := 0
	n1 := 0
	n2 := 0
	w1 := 0
	w2 := 0
	c1 := 0
	//	c2 := 0
	ntotal := 0
	nin := 0
	nout := 0
	offs := 0
	f := 0.0
	df := 0.0
	d2f := 0.0
	v := 0.0
	s := 0.0
	fown := 0.0
	deown := 0.0
	net := 0.0
	//	lnnet := 0.0
	mx := 0.0
	var bflag bool
	istart := 0
	//	ineurons := 0
	idfdnet := 0
	iderror := 0
	izeros := 0
	i_ := 0
	i1_ := 0

	//
	// Read network geometry, prepare data
	//
	nin = network.StructInfo[1]
	nout = network.StructInfo[2]
	ntotal = network.StructInfo[3]
	istart = network.StructInfo[5]
	c1 = cstart
	//	c2 = cstart + csize - 1
	//	ineurons = 0
	idfdnet = ntotal
	iderror = 2 * ntotal
	izeros = 3 * ntotal
	for j = 0; j <= csize - 1; j++ {
		network.Chunks[izeros][j] = 0
	}

	//
	// Forward pass:
	// 1. Load inputs from XY to Chunks[0:NIn-1,0:CSize-1]
	// 2. Forward pass
	//
	for i = 0; i <= nin - 1; i++ {
		for j = 0; j <= csize - 1; j++ {
			if network.ColumnSigmas[i] != 0.0 {
				network.Chunks[i][j] = (xy[c1 + j][i] - network.ColumnMeans[i]) / network.ColumnSigmas[i]
			}else {
				network.Chunks[i][j] = xy[c1 + j][i] - network.ColumnMeans[i]
			}
		}
	}
	for i = 0; i <= ntotal - 1; i++ {
		offs = istart + i * nfieldwidth
		if (network.StructInfo[offs + 0] > 0) || (network.StructInfo[offs + 0] == -5) {
			//
			// Activation function:
			// * calculate F vector, F(i) = F(NET(i))
			//
			n1 = network.StructInfo[offs + 2]
			for i_ = 0; i_ <= csize - 1; i_++ {
				network.Chunks[i][i_] = network.Chunks[n1][i_]
			}
			for j = 0; j <= csize - 1; j++ {
				MlpActivationFunction(network.Chunks[i][j], network.StructInfo[offs + 0], &f, &df, &d2f)
				network.Chunks[i][ j] = f
				network.Chunks[idfdnet + i][ j] = df
			}
			continue
		}
		if network.StructInfo[offs + 0] == 0 {
			//
			// Adaptive summator:
			// * calculate NET vector, NET(i) = SUM(W(j,i)*Neurons(j),j=N1..N2)
			//
			n1 = network.StructInfo[offs + 2]
			n2 = n1 + network.StructInfo[offs + 1] - 1
			w1 = network.StructInfo[offs + 3]
			w2 = w1 + network.StructInfo[offs + 1] - 1
			for i_ = 0; i_ <= csize - 1; i_++ {
				network.Chunks[i][ i_] = network.Chunks[izeros][ i_]
			}
			for j = n1; j <= n2; j++ {
				v = network.Weights[w1 + j - n1]
				for i_ = 0; i_ <= csize - 1; i_++ {
					network.Chunks[i][ i_] = network.Chunks[i][ i_] + v * network.Chunks[j][ i_]
				}
			}
			continue;
		}
		if network.StructInfo[offs + 0] < 0 {
			bflag = false
			if network.StructInfo[offs + 0] == -2 {
				//
				// input neuron, left unchanged
				//
				bflag = true
			}
			if network.StructInfo[offs + 0] == -3 {
				//
				// "-1" neuron
				//
				for k = 0; k <= csize - 1; k++ {
					network.Chunks[i][ k] = -1
				}
				bflag = true
			}
			if network.StructInfo[offs + 0] == -4 {
				//
				// "0" neuron
				//
				for k = 0; k <= csize - 1; k++ {
					network.Chunks[i][ k] = 0
				}
				bflag = true
			}

			if !(bflag) {
				return fmt.Errorf("MLPChunkedGradient: internal error - unknown neuron type!")
			}
			continue
		}
	}
	//
	// Post-processing, error, dError/dOut
	//
	for i = 0; i <= ntotal - 1; i++ {
		for i_ = 0; i_ <= csize - 1; i_++ {
			network.Chunks[iderror + i][ i_] = network.Chunks[izeros][ i_]
		}
	}
	if !((network.StructInfo[6] == 0) || (network.StructInfo[6] == 1)) {
		return fmt.Errorf("MLPChunkedGradient: unknown normalization type!")
	}
	if network.StructInfo[6] == 1 {
		//
		// Softmax output, classification network.
		//
		// For each K = 0..CSize-1 do:
		// 1. place exp(outputs[k]) to NWBuf[0:NOut-1]
		// 2. place sum(exp(..)) to NET
		// 3. calculate dError/dOut and place it to the second block of Chunks
		//
		for k = 0; k <= csize - 1; k++ {
			//
			// Normalize
			//
			mx = network.Chunks[ntotal - nout][ k]
			for i = 1; i <= nout - 1; i++ {
				mx = math.Max(mx, network.Chunks[ntotal - nout + i][ k])
			}
			net = 0
			for i = 0; i <= nout - 1; i++ {
				network.NwBuf[i] = math.Exp(network.Chunks[ntotal - nout + i][ k] - mx)
				net = net + network.NwBuf[i]
			}

			//
			// Calculate error function and dError/dOut
			//
			if naturalerrorfunc {
				//
				// Natural error func.
				//
				//
				s = 1;
				//				lnnet = math.Log(net)
				kl = utils.RoundInt(xy[cstart + k][ nin])
				for i = 0; i <= nout - 1; i++ {
					if i == kl {
						v = 1
					}else {
						v = 0
					}
					network.Chunks[iderror + ntotal - nout + i][ k] = s * network.NwBuf[i] / net - v
					*e = *e + safecrossentropy(v, network.NwBuf[i] / net)
				}
			}else {
				//
				// Least squares error func
				// Error, dError/dOut(normalized)
				//
				kl = utils.RoundInt(xy[cstart + k][ nin])
				for i = 0; i <= nout - 1; i++ {
					if i == kl {
						v = network.NwBuf[i] / net - 1
					}else {
						v = network.NwBuf[i] / net
					}
					network.NwBuf[nout + i] = v
					*e += utils.SqrFloat64(v) / 2
				}

				//
				// From dError/dOut(normalized) to dError/dOut(non-normalized)
				//
				i1_ = (0) - (nout)
				v = 0.0
				for i_ = nout; i_ <= 2 * nout - 1; i_++ {
					v += network.NwBuf[i_] * network.NwBuf[i_ + i1_]
				}
				for i = 0; i <= nout - 1; i++ {
					fown = network.NwBuf[i]
					deown = network.NwBuf[nout + i]
					network.Chunks[iderror + ntotal - nout + i][ k] = (-v + deown * fown + deown * (net - fown)) * fown / utils.SqrFloat64(net)
				}
			}
		}
	}else {
		//
		// Normal output, regression network
		//
		// For each K = 0..CSize-1 do:
		// 1. calculate dError/dOut and place it to the second block of Chunks
		//
		for i = 0; i <= nout - 1; i++ {
			for j = 0; j <= csize - 1; j++ {
				v = network.Chunks[ntotal - nout + i][ j] * network.ColumnSigmas[nin + i] + network.ColumnMeans[nin + i] - xy[cstart + j][ nin + i]
				network.Chunks[iderror + ntotal - nout + i][ j] = v * network.ColumnSigmas[nin + i]
				*e += utils.SqrFloat64(v) / 2
			}
		}
	}


	//
	// Backpropagation
	//
	for i = ntotal - 1; i >= 0; i-- {
		//
		// Extract info
		//
		offs = istart + i * nfieldwidth
		if ( network.StructInfo[offs + 0] > 0 ) || (network.StructInfo[offs + 0] == -5 ) {
			//
			// Activation function
			//
			n1 = network.StructInfo[offs + 2]
			for k = 0; k <= csize - 1; k++ {
				network.Chunks[iderror + i][ k] = network.Chunks[iderror + i][ k] * network.Chunks[idfdnet + i][k]
			}
			for i_ = 0; i_ <= csize - 1; i_++ {
				network.Chunks[iderror + n1][ i_] = network.Chunks[iderror + n1][ i_] + network.Chunks[iderror + i][ i_]
			}
			continue
		}
		if network.StructInfo[offs + 0] == 0 {
			//
			// "Normal" activation function
			//
			n1 = network.StructInfo[offs + 2]
			n2 = n1 + network.StructInfo[offs + 1] - 1
			w1 = network.StructInfo[offs + 3]
			w2 = w1 + network.StructInfo[offs + 1] - 1
			for j = w1; j <= w2; j++ {
				v = 0.0
				for i_ = 0; i_ <= csize - 1; i_++ {
					v += network.Chunks[n1 + j - w1][ i_] * network.Chunks[iderror + i][ i_]
				}
				(*grad)[j] = (*grad)[j] + v
			}
			for j = n1; j <= n2; j++ {
				v = network.Weights[w1 + j - n1]
				for i_ = 0; i_ <= csize - 1; i_++ {
					network.Chunks[iderror + j][ i_] = network.Chunks[iderror + j][ i_] + v * network.Chunks[iderror + i][ i_]
				}
			}
			continue
		}
		if network.StructInfo[offs + 0] < 0 {
			bflag = false
			if (network.StructInfo[offs + 0] == -2 || network.StructInfo[offs + 0] == -3) || network.StructInfo[offs + 0] == -4 {
				//
				// Special neuron type, no back-propagation required
				//
				bflag = true;
			}
			if !(bflag) {
				return fmt.Errorf("MLPInternalCalculateGradient: unknown neuron type!")
			}
			continue
		}
	}
	return nil
}

/*************************************************************************
Internal subroutine for Hessian calculation.

WARNING!!! Unspeakable math far beyong human capabilities :)
*************************************************************************/
func mlphessianbatchinternal(network *Multilayerperceptron, xy *[][]float64, ssize int, naturalerr bool, e *float64, grad *[]float64, h *[][]float64) error {
	nin := 0;
	nout := 0
	wcount := 0
	ntotal := 0
	istart := 0
	i := 0
	j := 0
	k := 0
	kl := 0
	offs := 0
	n1 := 0
	n2 := 0
	w1 := 0
	w2 := 0
	s := 0.0
	t := 0.0
	v := 0.0
	et := 0.0
	f := 0.0
	df := 0.0
	d2f := 0.0
	deidyj := 0.0
	mx := 0.0
	q := 0.0
	z := 0.0
	s2 := 0.0
	expi := 0.0
	expj := 0.0
	i_ := 0
	i1_ := 0

	MlpProperties(network, &nin, &nout, &wcount)
	ntotal = network.StructInfo[3]
	istart = network.StructInfo[5]

	//
	// Prepare
	//
	x := make([]float64, nin - 1 + 1)
	desiredy := make([]float64, nout - 1 + 1)
	zeros := make([]float64, wcount - 1 + 1)
	gt := make([]float64, wcount - 1 + 1)
	rx := utils.MakeMatrixFloat64(ntotal + nout - 1 + 1, wcount - 1 + 1)
	ry := utils.MakeMatrixFloat64(ntotal + nout - 1 + 1, wcount - 1 + 1)
	rdx := utils.MakeMatrixFloat64(ntotal + nout - 1 + 1, wcount - 1 + 1)
	rdy := utils.MakeMatrixFloat64(ntotal + nout - 1 + 1, wcount - 1 + 1)
	*e = 0.0

	for i := 0; i <= wcount - 1; i++ {
		zeros[i] = 0
	}
	for i_ := 0; i_ <= wcount - 1; i_++ {
		(*grad)[i_] = zeros[i_]
	}
	for i := 0; i <= wcount - 1; i++ {
		for i_ := 0; i_ <= wcount - 1; i_++ {
			(*h)[i][ i_] = zeros[i_]
		}
	}

	//
	// Process
	//
	for k = 0; k <= ssize - 1; k++ {
		//
		// Process vector with MLPGradN.
		// Now Neurons, DFDNET and DError contains results of the last run.
		//
		for i_ = 0; i_ <= nin - 1; i_++ {
			x[i_] = (*xy)[k][ i_]
		}
		if MlpIsSoftMax(network) {
			//
			// class labels outputs
			//
			kl = utils.RoundInt((*xy)[k][ nin])
			for i = 0; i <= nout - 1; i++ {
				if i == kl {
					desiredy[i] = 1
				}    else {
					desiredy[i] = 0
				}
			}
		}else {
			//
			// real outputs
			//
			i1_ = (nin) - (0)
			for i_ = 0; i_ <= nout - 1; i_++ {
				desiredy[i_] = (*xy)[k][ i_ + i1_]
			}
		}
		if naturalerr {
			MlpGradn(network, &x, &desiredy, &et, &gt)
		}else {
			MlpGrad(network, &x, &desiredy, &et, &gt)
		}

		//
		// grad, error
		//
		*e += et;
		for i_ = 0; i_ <= wcount - 1; i_++ {
			(*grad)[i_] += gt[i_]
		}

		//
		// Hessian.
		// Forward pass of the R-algorithm
		//
		for i = 0; i <= ntotal - 1; i++ {
			offs = istart + i * nfieldwidth
			for i_ = 0; i_ <= wcount - 1; i_++ {
				rx[i][ i_] = zeros[i_]
			}
			for i_ = 0; i_ <= wcount - 1; i_++ {
				ry[i][ i_] = zeros[i_]
			}
			if network.StructInfo[offs + 0] > 0 || network.StructInfo[offs + 0] == -5 {
				//
				// Activation function
				//
				n1 = network.StructInfo[offs + 2]
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rx[i][ i_] = ry[n1][ i_]
				}
				v = network.DfdNet[i]
				for i_ = 0; i_ <= wcount - 1; i_++ {
					ry[i][ i_] = v * rx[i][ i_]
				}
				continue
			}
			if network.StructInfo[offs + 0] == 0 {
				//
				// Adaptive summator
				//
				n1 = network.StructInfo[offs + 2];
				n2 = n1 + network.StructInfo[offs + 1] - 1;
				w1 = network.StructInfo[offs + 3];
				w2 = w1 + network.StructInfo[offs + 1] - 1;
				for j = n1; j <= n2; j++ {
					v = network.Weights[w1 + j - n1]
					for i_ = 0; i_ <= wcount - 1; i_++ {
						rx[i][ i_] = rx[i][ i_] + v * ry[j][i_]
					}
					rx[i][ w1 + j - n1] = rx[i][ w1 + j - n1] + network.Neurons[j]
				}
				for i_ = 0; i_ <= wcount - 1; i_++ {
					ry[i][ i_] = rx[i][ i_]
				}
				continue
			}
			if network.StructInfo[offs + 0] < 0 {
				bflag := true
				if network.StructInfo[offs + 0] == -2 {
					//
					// input neuron, left unchanged
					//
					bflag = false
				}
				if network.StructInfo[offs + 0] == -3 {
					//
					// "-1" neuron, left unchanged
					//
					bflag = false
				}
				if network.StructInfo[offs + 0] == -4 {
					//
					// "0" neuron, left unchanged
					//
					bflag = false
				}
				if !(!bflag) {
					return fmt.Errorf("MLPHessianNBatch: internal error - unknown neuron type!")
				}
				continue
			}
		}

		//
		// Hessian. Backward pass of the R-algorithm.
		//
		// Stage 1. Initialize RDY
		//
		for i = 0; i <= ntotal + nout - 1; i++ {
			for i_ = 0; i_ <= wcount - 1; i_++ {
				rdy[i][ i_] = zeros[i_]
			}
		}
		if network.StructInfo[6] == 0 {
			//
			// Standardisation.
			//
			// In context of the Hessian calculation standardisation
			// is considered as additional layer with weightless
			// activation function:
			//
			// F(NET) := Sigma*NET
			//
			// So we add one more layer to forward pass, and
			// make forward/backward pass through this layer.
			//
			for i = 0; i <= nout - 1; i++ {
				n1 = ntotal - nout + i
				n2 = ntotal + i

				//
				// Forward pass from N1 to N2
				//
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rx[n2][ i_] = ry[n1][ i_]
				}
				v = network.ColumnSigmas[nin + i]
				for i_ = 0; i_ <= wcount - 1; i_++ {
					ry[n2][ i_] = v * rx[n2][ i_]
				}

				//
				// Initialization of RDY
				//
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdy[n2][ i_] = ry[n2][ i_]
				}

				//
				// Backward pass from N2 to N1:
				// 1. Calculate R(dE/dX).
				// 2. No R(dE/dWij) is needed since weight of activation neuron
				//    is fixed to 1. So we can update R(dE/dY) for
				//    the connected neuron (note that Vij=0, Wij=1)
				//
				df = network.ColumnSigmas[nin + i]
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdx[n2][ i_] = df * rdy[n2][ i_]
				}
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdy[n1][ i_] = rdy[n1][ i_] + rdx[n2][ i_]
				}
			}
		}else {
			//
			// Softmax.
			//
			// Initialize RDY using generalized expression for ei'(yi)
			// (see expression (9) from p. 5 of "Fast Exact Multiplication by the Hessian").
			//
			// When we are working with softmax network, generalized
			// expression for ei'(yi) is used because softmax
			// normalization leads to ei, which depends on all y's
			//
			if naturalerr {
				//
				// softmax + cross-entropy.
				// We have:
				//
				// S = sum(exp(yk)),
				// ei = sum(trn)*exp(yi)/S-trn_i
				//
				// j=i:   d(ei)/d(yj) = T*exp(yi)*(S-exp(yi))/S^2
				// j<>i:  d(ei)/d(yj) = -T*exp(yi)*exp(yj)/S^2
				//
				t = 0
				for i = 0; i <= nout - 1; i++ {
					t = t + desiredy[i]
				}
				mx = network.Neurons[ntotal - nout]
				for i = 0; i <= nout - 1; i++ {
					mx = math.Max(mx, network.Neurons[ntotal - nout + i])
				}
				s = 0
				for i = 0; i <= nout - 1; i++ {
					network.NwBuf[i] = math.Exp(network.Neurons[ntotal - nout + i] - mx)
					s = s + network.NwBuf[i]
				}
				for i = 0; i <= nout - 1; i++ {
					for j = 0; j <= nout - 1; j++ {
						if j == i {
							deidyj = t * network.NwBuf[i] * (s - network.NwBuf[i]) / utils.SqrFloat64(s)
							for i_ = 0; i_ <= wcount - 1; i_++ {
								rdy[ntotal - nout + i][ i_] = rdy[ntotal - nout + i][ i_] + deidyj * ry[ntotal - nout + i][ i_]
							}
						}else {
							deidyj = -(t * network.NwBuf[i] * network.NwBuf[j] / utils.SqrFloat64(s))
							for i_ = 0; i_ <= wcount - 1; i_++ {
								rdy[ntotal - nout + i][ i_] = rdy[ntotal - nout + i][ i_] + deidyj * ry[ntotal - nout + j][ i_]
							}
						}
					}
				}
			}else {

				//
				// For a softmax + squared error we have expression
				// far beyond human imagination so we dont even try
				// to comment on it. Just enjoy the code...
				//
				// P.S. That's why "natural error" is called "natural" -
				// compact beatiful expressions, fast code....
				//
				mx = network.Neurons[ntotal - nout]
				for i = 0; i <= nout - 1; i++ {
					mx = math.Max(mx, network.Neurons[ntotal - nout + i])
				}
				s = 0
				s2 = 0
				for i = 0; i <= nout - 1; i++ {
					network.NwBuf[i] = math.Exp(network.Neurons[ntotal - nout + i] - mx)
					s += network.NwBuf[i]
					s2 += utils.SqrFloat64(network.NwBuf[i])
				}
				q = 0
				for i = 0; i <= nout - 1; i++ {
					q = q + (network.Y[i] - desiredy[i]) * network.NwBuf[i]
				}
				for i = 0; i <= nout - 1; i++ {
					z = -q + (network.Y[i] - desiredy[i]) * s
					expi = network.NwBuf[i]
					for j = 0; j <= nout - 1; j++ {
						expj = network.NwBuf[j]
						if j == i {
							deidyj = expi / utils.SqrFloat64(s) * ((z + expi) * (s - 2 * expi) / s + expi * s2 / utils.SqrFloat64(s))
						}else {
							deidyj = expi * expj / utils.SqrFloat64(s) * (s2 / utils.SqrFloat64(s) - 2 * z / s - (expi + expj) / s + (network.Y[i] - desiredy[i]) - (network.Y[j] - desiredy[j]))
						}
						for i_ = 0; i_ <= wcount - 1; i_++ {
							rdy[ntotal - nout + i][ i_] = rdy[ntotal - nout + i][ i_] + deidyj * ry[ntotal - nout + j][ i_]
						}
					}
				}
			}
		}

		//
		// Hessian. Backward pass of the R-algorithm
		//
		// Stage 2. Process.
		//
		for i = ntotal - 1; i >= 0; i-- {
			//
			// Possible variants:
			// 1. Activation function
			// 2. Adaptive summator
			// 3. Special neuron
			//
			offs = istart + i * nfieldwidth
			if network.StructInfo[offs + 0] > 0 || network.StructInfo[offs + 0] == -5 {
				n1 = network.StructInfo[offs + 2]

				//
				// First, calculate R(dE/dX).
				//
				MlpActivationFunction(network.Neurons[n1], network.StructInfo[offs + 0], &f, &df, &d2f)
				v = d2f * network.DError[i]
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdx[i][i_] = df * rdy[i][ i_]
				}
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdx[i][ i_] = rdx[i][ i_] + v * rx[i][i_]
				}

				//
				// No R(dE/dWij) is needed since weight of activation neuron
				// is fixed to 1.
				//
				// So we can update R(dE/dY) for the connected neuron.
				// (note that Vij=0, Wij=1)
				//
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdy[n1][ i_] = rdy[n1][ i_] + rdx[i][ i_]
				}
				continue
			}
			if network.StructInfo[offs + 0] == 0 {
				//
				// Adaptive summator
				//
				n1 = network.StructInfo[offs + 2]
				n2 = n1 + network.StructInfo[offs + 1] - 1
				w1 = network.StructInfo[offs + 3]
				w2 = w1 + network.StructInfo[offs + 1] - 1

				//
				// First, calculate R(dE/dX).
				//
				for i_ = 0; i_ <= wcount - 1; i_++ {
					rdx[i][ i_] = rdy[i][ i_]
				}

				//
				// Then, calculate R(dE/dWij)
				//
				for j = w1; j <= w2; j++ {
					v = network.Neurons[n1 + j - w1]
					for i_ = 0; i_ <= wcount - 1; i_++ {
						(*h)[j][ i_] += v * rdx[i][i_]
					}
					v = network.DError[i]
					for i_ = 0; i_ <= wcount - 1; i_++ {
						(*h)[j][ i_] += v * ry[n1 + j - w1][ i_]
					}
				}

				//
				// And finally, update R(dE/dY) for connected neurons.
				//
				for j = w1; j <= w2; j++ {
					v = network.Weights[j]
					for i_ = 0; i_ <= wcount - 1; i_++ {
						rdy[n1 + j - w1][ i_] = rdy[n1 + j - w1][ i_] + v * rdx[i][ i_]
					}
					rdy[n1 + j - w1][ j] = rdy[n1 + j - w1][j] + network.DError[i]
				}
				continue
			}
			if network.StructInfo[offs + 0] < 0 {
				bflag := false
				if (network.StructInfo[offs + 0] == -2 || network.StructInfo[offs + 0] == -3) || network.StructInfo[offs + 0] == -4 {
					//
					// Special neuron type, no back-propagation required
					//
					bflag = true
				}
				if !(bflag) {
					return fmt.Errorf("MLPHessianNBatch: unknown neuron type!")
				}
				continue;

			}
		}
	}
	return nil
}

