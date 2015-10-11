package mlptrain
import "pr.optima/src/core/neural/mlpbase"

const(
	mindecay = 0.001
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
func MlpTrainLm(network *mlpbase.Multilayerperceptron ,xy *[][]float64,npoints int,decay float64,restarts int,info *int,rep *MlpReport ){
nin := 0
nout := 0
wcount := 0
lmftol := 0.0
lmsteptol := 0.0
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
lambdaup := 0.0
lambdadown := 0.0
minlbfgs.minlbfgsreport internalrep = new minlbfgs.minlbfgsreport();
minlbfgs.minlbfgsstate state = new minlbfgs.minlbfgsstate();
double[] x = new double[0];
double[] y = new double[0];
double[] wbase = new double[0];
double[] wdir = new double[0];
double[] wt = new double[0];
double[] wx = new double[0];
int pass = 0;
double[] wbest = new double[0];
double ebest = 0;
int invinfo = 0;
matinv.matinvreport invrep = new matinv.matinvreport();
int solverinfo = 0;
densesolver.densesolverreport solverrep = new densesolver.densesolverreport();
int i_ = 0;

info = 0;

mlpbase.mlpproperties(network, ref nin, ref nout, ref wcount);
lambdaup = 10;
lambdadown = 0.3;
lmftol = 0.001;
lmsteptol = 0.001;

//
// Test for inputs
//
if( npoints<=0 | restarts<1 )
{
info = -1;
return;
}
if( mlpbase.mlpissoftmax(network) )
{
for(i=0; i<=npoints-1; i++)
{
if( (int)Math.Round(xy[i,nin])<0 | (int)Math.Round(xy[i,nin])>=nout )
{
info = -2;
return;
}
}
}
decay = Math.Max(decay, mindecay);
info = 2;

//
// Initialize data
//
rep.ngrad = 0;
rep.nhess = 0;
rep.ncholesky = 0;

//
// General case.
// Prepare task and network. Allocate space.
//
mlpbase.mlpinitpreprocessor(network, xy, npoints);
g = new double[wcount-1+1];
h = new double[wcount-1+1, wcount-1+1];
hmod = new double[wcount-1+1, wcount-1+1];
wbase = new double[wcount-1+1];
wdir = new double[wcount-1+1];
wbest = new double[wcount-1+1];
wt = new double[wcount-1+1];
wx = new double[wcount-1+1];
ebest = math.maxrealnumber;

//
// Multiple passes
//
for(pass=1; pass<=restarts; pass++)
{

//
// Initialize weights
//
mlpbase.mlprandomize(network);

//
// First stage of the hybrid algorithm: LBFGS
//
for(i_=0; i_<=wcount-1;i_++)
{
wbase[i_] = network.weights[i_];
}
minlbfgs.minlbfgscreate(wcount, Math.Min(wcount, 5), wbase, state);
minlbfgs.minlbfgssetcond(state, 0, 0, 0, Math.Max(25, wcount));
while( minlbfgs.minlbfgsiteration(state) )
{

//
// gradient
//
for(i_=0; i_<=wcount-1;i_++)
{
network.weights[i_] = state.x[i_];
}
mlpbase.mlpgradbatch(network, xy, npoints, ref state.f, ref state.g);

//
// weight decay
//
v = 0.0;
for(i_=0; i_<=wcount-1;i_++)
{
v += network.weights[i_]*network.weights[i_];
}
state.f = state.f+0.5*decay*v;
for(i_=0; i_<=wcount-1;i_++)
{
state.g[i_] = state.g[i_] + decay*network.weights[i_];
}

//
// next iteration
//
rep.ngrad = rep.ngrad+1;
}
minlbfgs.minlbfgsresults(state, ref wbase, internalrep);
for(i_=0; i_<=wcount-1;i_++)
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
for(i_=0; i_<=wcount-1;i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = e+0.5*decay*v;
for(i_=0; i_<=wcount-1;i_++)
{
g[i_] = g[i_] + decay*network.weights[i_];
}
for(k=0; k<=wcount-1; k++)
{
h[k,k] = h[k,k]+decay;
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
for(i=0; i<=wcount-1; i++)
{
for(i_=0; i_<=wcount-1;i_++)
{
hmod[i,i_] = h[i,i_];
}
hmod[i,i] = hmod[i,i]+lambdav;
}
spd = trfac.spdmatrixcholesky(ref hmod, wcount, true);
rep.ncholesky = rep.ncholesky+1;
if( !spd )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}
densesolver.spdmatrixcholeskysolve(hmod, wcount, true, g, ref solverinfo, solverrep, ref wdir);
if( solverinfo<0 )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}
for(i_=0; i_<=wcount-1;i_++)
{
wdir[i_] = -1*wdir[i_];
}

//
// Lambda found.
// 1. Save old w in WBase
// 1. Test some stopping criterions
// 2. If error(w+wdir)>error(w), increase lambda
//
for(i_=0; i_<=wcount-1;i_++)
{
network.weights[i_] = network.weights[i_] + wdir[i_];
}
xnorm2 = 0.0;
for(i_=0; i_<=wcount-1;i_++)
{
xnorm2 += network.weights[i_]*network.weights[i_];
}
stepnorm = 0.0;
for(i_=0; i_<=wcount-1;i_++)
{
stepnorm += wdir[i_]*wdir[i_];
}
stepnorm = Math.Sqrt(stepnorm);
enew = mlpbase.mlperror(network, xy, npoints)+0.5*decay*xnorm2;
if( (double)(stepnorm)<(double)(lmsteptol*(1+Math.Sqrt(xnorm2))) )
{
break;
}
if( (double)(enew)>(double)(e) )
{
lambdav = lambdav*lambdaup*nu;
nu = nu*2;
continue;
}

//
// Optimize using inv(cholesky(H)) as preconditioner
//
matinv.rmatrixtrinverse(ref hmod, wcount, true, false, ref invinfo, invrep);
if( invinfo<=0 )
{

//
// if matrix can't be inverted then exit with errors
// TODO: make WCount steps in direction suggested by HMod
//
info = -9;
return;
}
for(i_=0; i_<=wcount-1;i_++)
{
wbase[i_] = network.weights[i_];
}
for(i=0; i<=wcount-1; i++)
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
for(i=0; i<=wcount-1; i++)
{
v = 0.0;
for(i_=i; i_<=wcount-1;i_++)
{
v += state.x[i_]*hmod[i,i_];
}
network.weights[i] = wbase[i]+v;
}
mlpbase.mlpgradbatch(network, xy, npoints, ref state.f, ref g);
for(i=0; i<=wcount-1; i++)
{
state.g[i] = 0;
}
for(i=0; i<=wcount-1; i++)
{
v = g[i];
for(i_=i; i_<=wcount-1;i_++)
{
state.g[i_] = state.g[i_] + v*hmod[i,i_];
}
}

//
// weight decay
// grad(x'*x) = A'*(x0+A*t)
//
v = 0.0;
for(i_=0; i_<=wcount-1;i_++)
{
v += network.weights[i_]*network.weights[i_];
}
state.f = state.f+0.5*decay*v;
for(i=0; i<=wcount-1; i++)
{
v = decay*network.weights[i];
for(i_=i; i_<=wcount-1;i_++)
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
for(i=0; i<=wcount-1; i++)
{
v = 0.0;
for(i_=i; i_<=wcount-1;i_++)
{
v += wt[i_]*hmod[i,i_];
}
network.weights[i] = wbase[i]+v;
}
mlpbase.mlphessianbatch(network, xy, npoints, ref e, ref g, ref h);
v = 0.0;
for(i_=0; i_<=wcount-1;i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = e+0.5*decay*v;
for(i_=0; i_<=wcount-1;i_++)
{
g[i_] = g[i_] + decay*network.weights[i_];
}
for(k=0; k<=wcount-1; k++)
{
h[k,k] = h[k,k]+decay;
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
for(i_=0; i_<=wcount-1;i_++)
{
v += network.weights[i_]*network.weights[i_];
}
e = 0.5*decay*v+mlpbase.mlperror(network, xy, npoints);
if( (double)(e)<(double)(ebest) )
{
ebest = e;
for(i_=0; i_<=wcount-1;i_++)
{
wbest[i_] = network.weights[i_];
}
}
}

//
// copy WBest to output
//
for(i_=0; i_<=wcount-1;i_++)
{
network.weights[i_] = wbest[i_];
}
}