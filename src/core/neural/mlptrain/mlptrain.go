package mlptrain
import (
	"pr.optima/src/core/neural/mlpbase"
	"math"
	"fmt"
)

const (
	mindecay = 0.001

	maxrealnumber = 1E300
	minrealnumber = 1E-300
)

/*************************************************************************
Training report:
	* NGrad     - number of gradient calculations
	* NHess     - number of Hessian calculations
	* NCholesky - number of Cholesky decompositions
*************************************************************************/
type MlpReport struct {
	NGrad     int
	NHess     int
	NCholesky int
}

/*************************************************************************
Cross-validation estimates of generalization error
*************************************************************************/
type MlpCvReport struct {
	RelclsError float64
	Avgce       float64
	RmsError    float64
	AvgError    float64
	AvgrelError float64
}


type  minlbfgsreport struct {
	iterationscount int
	nfev            int
	terminationtype int
}


/********************************************************************
reverse communication structure
********************************************************************/
type rcommstate struct {
	stage int
	ia    [...]int
	ba    [...]bool
	ra    [...]float64
	ca    [...]complex128
};

func NewRCommState() *rcommstate {
	return &rcommstate{
		stage : -1,
		ia : [0]int,
		ba : [0]bool,
		ra : [0]float64,
		ca : [0]complex128}
}


type linminstate struct {
	brackt bool
	stage1 bool
	infoc  int
	dg     float64
	dgm    float64
	dginit float64
	dgtest float64
	dgx    float64
	dgxm   float64
	dgy    float64
	dgym   float64
	finit  float64
	ftest1 float64
	fm     float64
	fx     float64
	fxm    float64
	fy     float64
	fym    float64
	stx    float64
	sty    float64
	stmin  float64
	stmax  float64
	width  float64
	width1 float64
	xtrapf float64
}

type minlbfgsstate struct {
	n                  int
	m                  int
	epsg               float64
	epsf               float64
	epsx               float64
	maxits             int
	xrep               bool
	stpmax             float64
	s                  []float64
	diffstep           float64
	nfev               int
	mcstage            int
	k                  int
	q                  int
	p                  int
	rho                [...]float64
	yk                 [...][...]float64
	sk                 [...][...]float64
	theta              [...]float64
	d                  [...]float64
	stp                float64
	work               [...]float64
	fold               float64
	trimthreshold      float64
	prectype           int
	gammak             float64
	denseh             [...][...]float64
	diagh              [...]float64
	fbase              float64
	fm2                float64
	fm1                float64;
	fp1                float64
	fp2                float64
	autobuf            [...]float64
	x                  [...]float64
	f                  float64
	g                  [...]float64
	needf              bool
	needfg             bool
	xupdated           bool
	rstate             *rcommstate
	repiterationscount int
	repnfev            int
	repterminationtype int
	lstate             *linminstate
}

func NewMinLbfgsState() *minlbfgsstate {
	return &minlbfgsstate{
		s :[0]float64,
		rho :[0]float64,
		yk :[0][ 0]float64,
		sk :[0][0]float64,
		theta :[0]float64,
		d :[0]float64,
		work :[0]float64,
		denseh :[0][ 0]float64,
		diagh :[0]float64,
		autobuf :[0]float64,
		x :[0]float64,
		g :[0]float64,
		rstate : NewRCommState(),
		lstate : &linminstate{}}
}

/*************************************************************************
Matrix inverse report:
* R1    reciprocal of condition number in 1-norm
* RInf  reciprocal of condition number in inf-norm
*************************************************************************/
type matinvreport struct {
	r1   float64
	rinf float64
}

type densesolverreport struct {
	r1   float64
	rinf float64
}


/*************************************************************************
		LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION

DESCRIPTION:
The subroutine minimizes function F(x) of N arguments by  using  a  quasi-
Newton method (LBFGS scheme) which is optimized to use  a  minimum  amount
of memory.
The subroutine generates the approximation of an inverse Hessian matrix by
using information about the last M steps of the algorithm  (instead of N).
It lessens a required amount of memory from a value  of  order  N^2  to  a
value of order 2*N*M.


REQUIREMENTS:
Algorithm will request following information during its operation:
* function value F and its gradient G (simultaneously) at given point X


USAGE:
1. User initializes algorithm state with MinLBFGSCreate() call
2. User tunes solver parameters with MinLBFGSSetCond() MinLBFGSSetStpMax()
   and other functions
3. User calls MinLBFGSOptimize() function which takes algorithm  state and
   pointer (delegate, etc.) to callback function which calculates F/G.
4. User calls MinLBFGSResults() to get solution
5. Optionally user may call MinLBFGSRestartFrom() to solve another problem
   with same N/M but another starting point and/or another function.
   MinLBFGSRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
	N       -   problem dimension. N>0
	M       -   number of corrections in the BFGS scheme of Hessian
				approximation update. Recommended value:  3<=M<=7. The smaller
				value causes worse convergence, the bigger will  not  cause  a
				considerably better convergence, but will cause a fall in  the
				performance. M<=N.
	X       -   initial solution approximation, array[0..N-1].


OUTPUT PARAMETERS:
	State   -   structure which stores algorithm state


NOTES:
1. you may tune stopping conditions with MinLBFGSSetCond() function
2. if target function contains exp() or other fast growing functions,  and
   optimization algorithm makes too large steps which leads  to  overflow,
   use MinLBFGSSetStpMax() function to bound algorithm's  steps.  However,
   L-BFGS rarely needs such a tuning.


  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgscreate(n, m int, x *[]float64, state *minlbfgsstate) error {
	if !(n >= 1) {
		return fmt.Errorf("MinLBFGSCreate: N<1!")
	}
	if !(m >= 1) {
		return fmt.Errorf("MinLBFGSCreate: M<1")
	}
	if !(m <= n) {
		return fmt.Errorf("MinLBFGSCreate: M>N")
	}
	if !(len(x) >= n) {
		return fmt.Errorf("MinLBFGSCreate: Length(X)<N!")
	}
	if res, _ := isfinitevector(x, n); !(res) {
		return fmt.Errorf("MinLBFGSCreate: N<1!")
	}
	minlbfgscreatex(n, m, x, 0, 0.0, state)
	return nil
}

/*************************************************************************
Extended subroutine for internal use only.

Accepts additional parameters:

	Flags - additional settings:
			* Flags = 0     means no additional settings
			* Flags = 1     "do not allocate memory". used when solving
							a many subsequent tasks with  same N/M  values.
							First  call MUST  be without this flag bit set,
							subsequent  calls   of   MinLBFGS   with   same
							MinLBFGSState structure can set Flags to 1.
	DiffStep - numerical differentiation step

  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgscreatex(n, m int, x *[]float64, flags int, diffstep float64, state *minlbfgsstate) {
	i := 0

	if !(n >= 1) {
		return fmt.Errorf("MinLBFGS: N too small!")
	}
	if !(m >= 1) {
		return fmt.Errorf("MinLBFGS: M too small!")
	}
	if !(m <= n) {
		return fmt.Errorf("MinLBFGS: M too large!")
	}

	//
	// Initialize
	//
	state.diffstep = diffstep
	state.n = n
	state.m = m
	allocatemem := flags % 2 == 0
	flags = flags / 2;
	if allocatemem {
		state.rho = [m]float64
		state.theta = [m]float64
		state.yk = [m][n]float64
		state.sk = [m][n]float64
		state.d = [n]float64
		state.x = [n]float64
		state.s = [n]float64
		state.g = [n]float64
		state.work = [n]float64
	}
	minlbfgssetcond(state, 0, 0, 0, 0)
	minlbfgssetxrep(state, false)
	minlbfgssetstpmax(state, 0)
	minlbfgsrestartfrom(state, x)
	for i = 0; i <= n - 1; i++ {
		state.s[i] = 1.0
	}
	state.prectype = 0
}
/*************************************************************************
Neural network training  using  modified  Levenberg-Marquardt  with  exact
Hessian calculation and regularization. Subroutine trains  neural  network
with restarts from random positions. Algorithm is well  suited  for  small
and medium scale problems (hundreds of weights).

INPUT PARAMETERS:
	Network     -   neural network with initialized geometry
	XY          -   training set
	NPoints     -   training set size
	Decay       -   weight decay constant, >=0.001
					Decay term 'Decay*||Weights||^2' is added to error
					function.
					If you don't know what Decay to choose, use 0.001.
	Restarts    -   number of restarts from random position, >0.
					If you don't know what Restarts to choose, use 2.

OUTPUT PARAMETERS:
	Network     -   trained neural network.
	Info        -   return code:
					* -9, if internal matrix inverse subroutine failed
					* -2, if there is a point with class number
						  outside of [0..NOut-1].
					* -1, if wrong parameters specified
						  (NPoints<0, Restarts<1).
					*  2, if task has been solved.
	Rep         -   training report

  -- ALGLIB --
	 Copyright 10.03.2009 by Bochkanov Sergey
*************************************************************************/
func MlpTrainLm(network *mlpbase.Multilayerperceptron, xy *[][]float64, npoints int, decay float64, restarts int, info *int, rep *MlpReport) {
	nin := 0
	nout := 0
	wcount := 0
	i := 0
	k := 0
	v := 0.0
	e := 0.0
	enew := 0.0
	xnorm2 := 0.0
	stepnorm := 0.0
	g := [0]float64
	d := [0]float64
	h := [0][0]float64
	hmod := [0][0]float64
	z := [0][0]float64
	nu := 0.0
	lambdav := 0.0
	internalrep := &minlbfgsreport{}
	state := NewMinLbfgsState()
	x := [0]float64
	y := [0]float64
	wbase := [0]float64
	wdir := [0]float64
	wt := [0]float64
	wx := [0]float64
	pass := 0
	wbest := [0]float64
	ebest := 0.0
	invinfo := 0
	invrep := &matinvreport{}
	solverinfo := 0
	solverrep := &densesolverreport{}
	i_ := 0

	info = 0

	mlpbase.MlpProperties(network, &nin, &nout, &wcount)
	lambdaup := 10.0
	lambdadown := 0.3
	lmftol := 0.001
	lmsteptol := 0.001

	//
	// Test for inputs
	//
	if npoints <= 0 | restarts < 1 {
		info = -1
		return
	}
	if mlpbase.MlpIsSoftMax(network) {
		for i = 0; i <= npoints - 1; i++ {
			if int(round(xy[i][ nin])) < 0 | int(round(xy[i][nin])) >= nout
			{
				info = -2
				return
			}
		}
	}
	decay = math.Max(decay, mindecay)
	info = 2

	//
	// Initialize data
	//
	rep.NGrad = 0
	rep.NHess = 0
	rep.NCholesky = 0

	//
	// General case.
	// Prepare task and network. Allocate space.
	//
	mlpbase.MlpInitPreprocessor(network, xy, npoints);
	g = [wcount - 1 + 1]float64
	h = [wcount - 1 + 1][ wcount - 1 + 1]float64
	hmod = [wcount - 1 + 1][ wcount - 1 + 1]float64
	wbase = [wcount - 1 + 1]float64
	wdir = [wcount - 1 + 1]float64
	wbest = [wcount - 1 + 1]float64
	wt = [wcount - 1 + 1]float64
	wx = [wcount - 1 + 1]float64
	ebest = maxrealnumber

	//
	// Multiple passes
	//
	for pass = 1; pass <= restarts; pass++ {
		//
		// Initialize weights
		//
		mlpbase.MlpRandomize(network)

		//
		// First stage of the hybrid algorithm: LBFGS
		//
		for i_ = 0; i_ <= wcount - 1; i_++ {
			wbase[i_] = network.Weights[i_]
		}
		minlbfgs.minlbfgscreate(wcount, Math.Min(wcount, 5), wbase, state);
		minlbfgs.minlbfgssetcond(state, 0, 0, 0, Math.Max(25, wcount));
		while(minlbfgs.minlbfgsiteration(state))
		{

			//
			// gradient
			//
			for (i_ = 0; i_ <= wcount - 1; i_++)
			{
			network.weights[i_] = state.x[i_];
			}
			mlpbase.mlpgradbatch(network, xy, npoints, ref state.f, ref state.g);

		//
		// weight decay
		//
		v = 0.0;
		for (i_ = 0; i_ <= wcount - 1; i_++)
		{
		v += network.weights[i_]*network.weights[i_];
		}
		state.f = state.f + 0.5 * decay * v;
		for (i_ = 0; i_ <= wcount - 1; i_++)
		{
		state.g[i_] = state.g[i_] + decay*network.weights[i_];
		}

		//
		// next iteration
		//
		rep.ngrad = rep.ngrad + 1;
	}
	minlbfgs.minlbfgsresults(state, ref wbase, internalrep);
for (i_ = 0; i_<=wcount-1; i_++)
{
network.weights[i_] = wbase[i_];
}

//
// Second stage of the hybrid algorithm: LM
//
// Initialize H with identity matrix,
// G with gradient,
// E with regularized error.
//
mlpbase.mlphessianbatch(network, xy, npoints, ref e, ref g, ref h);
v = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = e+0.5*decay*v;
for (i_ = 0; i_<=wcount-1; i_++)
{
g[i_] = g[i_] + decay*network.weights[i_];
}
for (k = 0; k<=wcount-1; k++)
{
h[k, k] = h[k, k]+decay;
}
rep.nhess = rep.nhess+1;
lambdav = 0.001;
nu = 2;
while( true )
{

//
// 1. HMod = H+lambda*I
// 2. Try to solve (H+Lambda*I)*dx = -g.
//    Increase lambda if left part is not positive definite.
//
for (i = 0; i<=wcount-1; i++)
{
for (i_ = 0; i_<=wcount-1; i_++)
{
hmod[i, i_] = h[i, i_];
}
hmod[i, i] = hmod[i, i]+lambdav;
}
spd = trfac.spdmatrixcholesky(ref hmod, wcount, true);
rep.ncholesky = rep.ncholesky+1;
if ( !spd )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}
densesolver.spdmatrixcholeskysolve(hmod, wcount, true, g, ref solverinfo, solverrep, ref wdir);
if ( solverinfo<0 )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}
for (i_ = 0; i_<=wcount-1; i_++)
{
wdir[i_] = -1*wdir[i_];
}

//
// Lambda found.
// 1. Save old w in WBase
// 1. Test some stopping criterions
// 2. If error(w+wdir)>error(w), increase lambda
//
for (i_ = 0; i_<=wcount-1; i_++)
{
network.weights[i_] = network.weights[i_] + wdir[i_];
}
xnorm2 = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
xnorm2 += network.weights[i_]*network.weights[i_];
}
stepnorm = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
stepnorm += wdir[i_]*wdir[i_];
}
stepnorm = Math.Sqrt(stepnorm);
enew = mlpbase.mlperror(network, xy, npoints)+0.5*decay*xnorm2;
if( (double)(stepnorm)<(double)(lmsteptol*(1+Math.Sqrt(xnorm2))) )
{
break;
}
if ( (double)(enew)>(double)(e) )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}

//
// Optimize using inv(cholesky(H)) as preconditioner
//
matinv.rmatrixtrinverse(ref hmod, wcount, true, false, ref invinfo, invrep);
if ( invinfo<=0 )
{

//
// if matrix can't be inverted then exit with errors
// TODO: make WCount steps in direction suggested by HMod
//
info = -9;
return;
}
for (i_ = 0; i_<=wcount-1; i_++)
{
wbase[i_] = network.weights[i_];
}
for (i = 0; i<=wcount-1; i++)
{
wt[i] = 0;
}
minlbfgs.minlbfgscreatex(wcount, wcount, wt, 1, 0.0, state);
minlbfgs.minlbfgssetcond(state, 0, 0, 0, 5);
while( minlbfgs.minlbfgsiteration(state) )
{

//
// gradient
//
for (i = 0; i<=wcount-1; i++)
{
v = 0.0;
for (i_ = i; i_<=wcount-1; i_++)
{
v += state.x[i_]*hmod[i, i_];
}
network.weights[i] = wbase[i]+v;
}
mlpbase.mlpgradbatch(network, xy, npoints, ref state.f, ref g);
for (i = 0; i<=wcount-1; i++)
{
state.g[i] = 0;
}
for (i = 0; i<=wcount-1; i++)
{
v = g[i];
for (i_= i; i_<=wcount-1; i_++)
{
state.g[i_] = state.g[i_] + v*hmod[i,i_];
}
}

//
// weight decay
// grad(x'*x) = A'*(x0+A*t)
//
v = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
v += network.weights[i_]*network.weights[i_];
}
state.f = state.f+0.5*decay*v;
for (i = 0; i<=wcount-1; i++)
{
v = decay*network.weights[i];
for (i_ = i; i_<=wcount-1; i_++)
{
state.g[i_] = state.g[i_] + v*hmod[i,i_];
}
}

//
// next iteration
//
rep.ngrad = rep.ngrad+1;
}
minlbfgs.minlbfgsresults(state, ref wt, internalrep);

//
// Accept new position.
// Calculate Hessian
//
for (i = 0; i<=wcount-1; i++)
{
v = 0.0;
for (i_ = i; i_<=wcount-1; i_++)
{
v += wt[i_]*hmod[i, i_];
}
network.weights[i] = wbase[i]+v;
}
mlpbase.mlphessianbatch(network, xy, npoints, ref e, ref g, ref h);
v = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = e+0.5*decay*v;
for (i_ = 0; i_<=wcount-1; i_++)
{
g[i_] = g[i_] + decay*network.weights[i_];
}
for (k = 0; k<=wcount-1; k++)
{
h[k, k] = h[k, k]+decay;
}
rep.nhess = rep.nhess+1;

//
// Update lambda
//
lambdav = lambdav*lambdadown;
nu = 2;
}

//
// update WBest
//
v = 0.0;
for (i_ = 0; i_<=wcount-1; i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = 0.5*decay*v+mlpbase.mlperror(network, xy, npoints);
if ( (double)(e)<(double)(ebest) )
{
ebest = e;
for (i_ = 0; i_<=wcount-1; i_++)
{
wbest[i_] = network.weights[i_];
}
}
}

//
// copy WBest to output
//
for (i_ = 0; i_ <= wcount - 1; i_++)
{
network.weights[i_] = wbest[i_];
}
}


/*************************************************************************
This function checks that all values from X[] are finite

  -- ALGLIB --
	 Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
func isfinitevector(double[] x, int n)(bool, error){
if !(n>=0){
return false fmt.Errorf("APSERVIsFiniteVector: internal error (N<0)")
}

for i := 0; i<=n-1; i++{
if !isfinite(x[i]) {
return false, nil
}
}

return true, nil
}

func round(f float64) float64 {
return math.Floor(f + .5)
}

func isfinite(d float64) bool {
return !math.IsNaN(d) && !(d > math.MaxFloat64 || d < -math.MaxFloat64)
}