package mlptrain
import (
	"math"
	"fmt"
	"math/cmplx"
	"math/rand"
	"pr.optima/src/core/neural/mlpbase"
	"core/neural/utils"
)

const (
	mindecay = 0.001

	maxrealnumber = 1E300
	minrealnumber = 1E-300
	machineepsilon = 5E-16

	stpmin = 1.0E-50
	defstpmax = 1.0E+50
	ftol = 0.001
	xtol = 100 * machineepsilon
	maxfev = 20
	gtol = 0.4
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
	ia    []int
	ba    []bool
	ra    []float64
	ca    []complex128
};

func NewRCommState() *rcommstate {
	return &rcommstate{
		stage : -1,
		ia : make([]int, 0),
		ba : make([]bool, 0),
		ra : make([]float64, 0),
		ca : make([]complex128, 0)}
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
	rho                []float64
	yk                 [][]float64
	sk                 [][]float64
	theta              []float64
	d                  []float64
	stp                float64
	work               []float64
	fold               float64
	trimthreshold      float64
	prectype           int
	gammak             float64
	denseh             [][]float64
	diagh              []float64
	fbase              float64
	fm2                float64
	fm1                float64
	fp1                float64
	fp2                float64
	autobuf            []float64
	x                  []float64
	f                  float64
	g                  []float64
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
		s :make([]float64, 0),
		rho :make([]float64, 0),
		yk :utils.MakeMatrixFloat64(0, 0),
		sk :utils.MakeMatrixFloat64(0, 0),
		theta :make([]float64, 0),
		d :make([]float64, 0),
		work :make([]float64, 0),
		denseh :utils.MakeMatrixFloat64(0, 0),
		diagh :make([]float64, 0),
		autobuf :make([]float64, 0),
		x :make([]float64, 0),
		g :make([]float64, 0),
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
	if !(len(*x) >= n) {
		return fmt.Errorf("MinLBFGSCreate: Length(X)<N!")
	}
	if res, _ := utils.IsFiniteVector(*x, n); !(res) {
		return fmt.Errorf("MinLBFGSCreate: N<1!")
	}
	return minlbfgscreatex(n, m, x, 0, 0.0, state)
}

/*************************************************************************
This function sets stopping conditions for L-BFGS optimization algorithm.

INPUT PARAMETERS:
	State   -   structure which stores algorithm state
	EpsG    -   >=0
				The  subroutine  finishes  its  work   if   the  condition
				|v|<EpsG is satisfied, where:
				* |.| means Euclidian norm
				* v - scaled gradient vector, v[i]=g[i]*s[i]
				* g - gradient
				* s - scaling coefficients set by MinLBFGSSetScale()
	EpsF    -   >=0
				The  subroutine  finishes  its work if on k+1-th iteration
				the  condition  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
				is satisfied.
	EpsX    -   >=0
				The subroutine finishes its work if  on  k+1-th  iteration
				the condition |v|<=EpsX is fulfilled, where:
				* |.| means Euclidian norm
				* v - scaled step vector, v[i]=dx[i]/s[i]
				* dx - ste pvector, dx=X(k+1)-X(k)
				* s - scaling coefficients set by MinLBFGSSetScale()
	MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
				iterations is unlimited.

Passing EpsG=0, EpsF=0, EpsX=0 and MaxIts=0 (simultaneously) will lead to
automatic stopping criterion selection (small EpsX).

  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgssetcond(state *minlbfgsstate, epsg, epsf, epsx float64, maxits int) error {
	if !(utils.IsFinite(epsg)) {
		fmt.Errorf("MinLBFGSSetCond: EpsG is not finite number!")
	}
	if !(epsg >= 0) {
		fmt.Errorf("MinLBFGSSetCond: negative EpsG!")
	}
	if !(utils.IsFinite(epsf)) {
		fmt.Errorf("MinLBFGSSetCond: EpsF is not finite number!")
	}
	if !(epsf >= 0) {
		fmt.Errorf("MinLBFGSSetCond: negative EpsF!")
	}
	if !(utils.IsFinite(epsx)) {
		fmt.Errorf("MinLBFGSSetCond: EpsX is not finite number!")
	}
	if !(epsx >= 0) {
		fmt.Errorf("MinLBFGSSetCond: negative EpsX!")
	}
	if !(maxits >= 0) {
		fmt.Errorf("MinLBFGSSetCond: negative MaxIts!")
	}

	if ((epsg == 0 && epsf == 0) && epsx == 0) && maxits == 0 {
		epsx = 1.0E-6
	}
	state.epsg = epsg
	state.epsf = epsf
	state.epsx = epsx
	state.maxits = maxits
	return nil
}

/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
	State   -   structure which stores algorithm state
	NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinLBFGSOptimize().


  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgssetxrep(state *minlbfgsstate, needxrep bool) {
	state.xrep = needxrep
}

/*************************************************************************
This function sets maximum step length

INPUT PARAMETERS:
	State   -   structure which stores algorithm state
	StpMax  -   maximum step length, >=0. Set StpMax to 0.0 (default),  if
				you don't want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  leads  to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgssetstpmax(state *minlbfgsstate, stpmax float64) error {
	if !(utils.IsFinite(stpmax)) {
		return fmt.Errorf("MinLBFGSSetStpMax: StpMax is not finite!")
	}
	if !(stpmax >= 0) {
		return fmt.Errorf("MinLBFGSSetStpMax: StpMax<0!")
	}
	state.stpmax = stpmax
	return nil
}

/*************************************************************************
Clears request fileds (to be sure that we don't forgot to clear something)
*************************************************************************/
func clearrequestfields(state *minlbfgsstate) {
	state.needf = false
	state.needfg = false
	state.xupdated = false
}

/*************************************************************************
This  subroutine restarts LBFGS algorithm from new point. All optimization
parameters are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
	State   -   structure used to store algorithm state
	X       -   new starting point.

  -- ALGLIB --
	 Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgsrestartfrom(state *minlbfgsstate, x *[]float64) error {
	if !(len(*x) >= state.n) {
		return fmt.Errorf("MinLBFGSRestartFrom: Length(X)<N!")
	}
	if res, err := utils.IsFiniteVector(*x, state.n); !(res) || err != nil {
		return fmt.Errorf("MinLBFGSRestartFrom: X contains infinite or NaN values!")
	}

	for i := 0; i <= state.n - 1; i++ {
		state.x[i] = (*x)[i]
	}
	state.rstate.ia = make([]int, 5 + 1)
	state.rstate.ra = make([]float64, 1 + 1)
	state.rstate.stage = -1
	clearrequestfields(state)
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
func minlbfgscreatex(n, m int, x *[]float64, flags int, diffstep float64, state *minlbfgsstate) error {
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
		state.rho = make([]float64, m)
		state.theta = make([]float64, m)
		state.yk = utils.MakeMatrixFloat64(m, n)
		state.sk = utils.MakeMatrixFloat64(m, n)
		state.d = make([]float64, n)
		state.x = make([]float64, n)
		state.s = make([]float64, n)
		state.g = make([]float64, n)
		state.work = make([]float64, n)
	}
	if err := minlbfgssetcond(state, 0, 0, 0, 0); err != nil {
		return err
	}
	minlbfgssetxrep(state, false)
	if err := minlbfgssetstpmax(state, 0); err != nil {
		return err
	}
	if err := minlbfgsrestartfrom(state, x); err != nil {
		return err
	}

	for i := 0; i <= n - 1; i++ {
		state.s[i] = 1.0
	}
	state.prectype = 0

	return nil
}

/*************************************************************************
This subroutine is used to prepare threshold value which will be used for
trimming of the target function (see comments on TrimFunction() for more
information).

This function accepts only one parameter: function value at the starting
point. It returns threshold which will be used for trimming.

  -- ALGLIB --
	 Copyright 10.05.2011 by Bochkanov Sergey
*************************************************************************/
func trimprepare(f float64, threshold *float64) {
	*threshold = 10 * (math.Abs(f) + 1)
}

/*************************************************************************
Normalizes direction/step pair: makes |D|=1, scales Stp.
If |D|=0, it returns, leavind D/Stp unchanged.

  -- ALGLIB --
	 Copyright 01.04.2010 by Bochkanov Sergey
*************************************************************************/
func linminnormalized(d *[]float64, stp *float64, n int) {
	mx := 0.0
	s := 0.0
	i := 0
	i_ := 0

	//
	// first, scale D to avoid underflow/overflow durng squaring
	//
	for i = 0; i <= n - 1; i++ {
		mx = math.Max(mx, math.Abs((*d)[i]))
	}
	if mx == 0 {
		return
	}
	s = 1 / mx
	for i_ = 0; i_ <= n - 1; i_++ {
		(*d)[i_] = s * (*d)[i_]
	}
	*stp = *stp / s

	//
	// normalize D
	//
	s = 0.0
	for i_ = 0; i_ <= n - 1; i_++ {
		s += (*d)[i_] * (*d)[i_]
	}
	s = 1 / math.Sqrt(s)
	for i_ = 0; i_ <= n - 1; i_++ {
		(*d)[i_] = s * (*d)[i_]
	}
	*stp = *stp / s
}

/*************************************************************************
Basic Cholesky solver for ScaleA*Cholesky(A)'*x = y.

This subroutine assumes that:
* A*ScaleA is well scaled
* A is well-conditioned, so no zero divisions or overflow may occur

INPUT PARAMETERS:
	CHA     -   Cholesky decomposition of A
	SqrtScaleA- square root of scale factor ScaleA
	N       -   matrix size
	IsUpper -   storage type
	XB      -   right part
	Tmp     -   buffer; function automatically allocates it, if it is  too
				small.  It  can  be  reused  if function is called several
				times.

OUTPUT PARAMETERS:
	XB      -   solution

NOTES: no assertion or tests are done during algorithm operation

  -- ALGLIB --
	 Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
func fblscholeskysolve(cha *[][]float64, sqrtscalea float64, n int, isupper bool, xb, tmp *[]float64) {
	i := 0
	v := 0.0
	i_ := 0

	if len(*tmp) < n {
		*tmp = make([]float64, n)
	}

	//
	// A = L*L' or A=U'*U
	//
	if isupper {
		//
		// Solve U'*y=b first.
		//
		for i = 0; i <= n - 1; i++ {
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
			if i < n - 1 {
				v = (*xb)[i]
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*xb)[i_] = (*xb)[i_] - v * (*tmp)[i_]
				}
			}
		}

		//
		// Solve U*x=y then.
		//
		for i = n - 1; i >= 0; i-- {
			if i < n - 1 {
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				v = 0.0
				for i_ = i + 1; i_ <= n - 1; i_++ {
					v += (*tmp)[i_] * (*xb)[i_]
				}
				(*xb)[i] = (*xb)[i] - v
			}
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
		}
	}else {
		//
		// Solve L*y=b first
		//
		for i = 0; i <= n - 1; i++ {
			if i > 0 {
				for i_ = 0; i_ <= i - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				v = 0.0
				for i_ = 0; i_ <= i - 1; i_++ {
					v += (*tmp)[i_] * (*xb)[i_]
				}
				(*xb)[i] = (*xb)[i] - v
			}
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
		}

		//
		// Solve L'*x=y then.
		//
		for i = n - 1; i >= 0; i-- {
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
			if i > 0 {
				v = (*xb)[i]
				for i_ = 0; i_ <= i - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				for i_ = 0; i_ <= i - 1; i_++ {
					(*xb)[i_] = (*xb)[i_] - v * (*tmp)[i_]
				}
			}
		}
	}
}

func mcstep(stx, fx, dx, sty, fy, dy, stp *float64, fp, dp float64, brackt *bool, stmin, stmax float64, info *int) {
	var bound bool
	gamma := 0.0
	p := 0.0
	q := 0.0
	r := 0.0
	s := 0.0
	sgnd := 0.0
	stpc := 0.0
	stpf := 0.0
	stpq := 0.0
	theta := 0.0

	*info = 0

	//
	//     CHECK THE INPUT PARAMETERS FOR ERRORS.
	//
	if ((*brackt && (*stp <= (math.Min(*stx, *sty)) || *stp >= (math.Max(*stx, *sty)))) || (*dx * (*stp - *stx)) >= 0) || stmax < stmin {
		return
	}

	//
	//     DETERMINE IF THE DERIVATIVES HAVE OPPOSITE SIGN.
	//
	sgnd = dp * (*dx / math.Abs(*dx))

	//
	//     FIRST CASE. A HIGHER FUNCTION VALUE.
	//     THE MINIMUM IS BRACKETED. IF THE CUBIC STEP IS CLOSER
	//     TO STX THAN THE QUADRATIC STEP, THE CUBIC STEP IS TAKEN,
	//     ELSE THE AVERAGE OF THE CUBIC AND QUADRATIC STEPS IS TAKEN.
	//
	if fp > *fx {
		*info = 1
		bound = true
		theta = 3 * (*fx - fp) / (*stp - *stx) + *dx + dp
		s = math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))
		_s := theta / s
		gamma = s * math.Sqrt((_s * _s) - *dx / s * (dp / s))
		if *stp < *stx {
			gamma = -gamma
		}
		p = gamma - *dx + theta
		q = gamma - *dx + gamma + dp
		r = p / q
		stpc = *stx + r * (*stp - *stx)
		stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2 * (*stp - *stx)
		if math.Abs(stpc - *stx) < math.Abs(stpq - *stx) {
			stpf = stpc
		}else {
			stpf = stpc + (stpq - stpc) / 2
		}
		*brackt = true
	}else {
		if sgnd < 0 {
			//
			//     SECOND CASE. A LOWER FUNCTION VALUE AND DERIVATIVES OF
			//     OPPOSITE SIGN. THE MINIMUM IS BRACKETED. IF THE CUBIC
			//     STEP IS CLOSER TO STX THAN THE QUADRATIC (SECANT) STEP,
			//     THE CUBIC STEP IS TAKEN, ELSE THE QUADRATIC STEP IS TAKEN.
			//
			*info = 2
			bound = false
			theta = 3 * (*fx - fp) / (*stp - *stx) + *dx + dp
			s = math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))
			_s := theta / s
			gamma = s * math.Sqrt((_s * _s) - *dx / s * (dp / s))
			if *stp > *stx {
				gamma = -gamma
			}
			p = gamma - dp + theta
			q = gamma - dp + gamma + *dx
			r = p / q
			stpc = *stp + r * (*stx - *stp)
			stpq = *stp + dp / (dp - *dx) * (*stx - *stp)
			if math.Abs(stpc - *stp) > math.Abs(stpq - *stp) {
				stpf = stpc
			}else {
				stpf = stpq
			}
			*brackt = true
		}else {
			if math.Abs(dp) < math.Abs(*dx) {
				//
				//     THIRD CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
				//     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DECREASES.
				//     THE CUBIC STEP IS ONLY USED IF THE CUBIC TENDS TO INFINITY
				//     IN THE DIRECTION OF THE STEP OR IF THE MINIMUM OF THE CUBIC
				//     IS BEYOND STP. OTHERWISE THE CUBIC STEP IS DEFINED TO BE
				//     EITHER STPMIN OR STPMAX. THE QUADRATIC (SECANT) STEP IS ALSO
				//     COMPUTED AND IF THE MINIMUM IS BRACKETED THEN THE THE STEP
				//     CLOSEST TO STX IS TAKEN, ELSE THE STEP FARTHEST AWAY IS TAKEN.
				//
				*info = 3
				bound = true
				theta = 3 * (*fx - fp) / (*stp - *stx) + *dx + dp
				s = math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))

				//
				//        THE CASE GAMMA = 0 ONLY ARISES IF THE CUBIC DOES NOT TEND
				//        TO INFINITY IN THE DIRECTION OF THE STEP.
				//
				_s := theta / s
				gamma = s * math.Sqrt(math.Max(0, (_s * _s) - *dx / s * (dp / s)))
				if *stp > *stx {
					gamma = -gamma
				}
				p = gamma - dp + theta
				q = gamma + (*dx - dp) + gamma
				r = p / q
				if r < 0 && gamma != 0 {
					stpc = *stp + r * (*stx - *stp)
				}else {
					if *stp > *stx {
						stpc = stmax
					}else {
						stpc = stmin
					}
				}
				stpq = *stp + dp / (dp - *dx) * (*stx - *stp)
				if *brackt {
					if math.Abs(*stp - stpc) < math.Abs(*stp - stpq) {
						stpf = stpc
					}else {
						stpf = stpq
					}
				}else {
					if math.Abs(*stp - stpc) > math.Abs(*stp - stpq) {
						stpf = stpc
					}else {
						stpf = stpq
					}
				}
			}else {
				//
				//     FOURTH CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
				//     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DOES
				//     NOT DECREASE. IF THE MINIMUM IS NOT BRACKETED, THE STEP
				//     IS EITHER STPMIN OR STPMAX, ELSE THE CUBIC STEP IS TAKEN.
				//
				*info = 4
				bound = false
				if *brackt {
					theta = 3 * (fp - *fy) / (*sty - *stp) + *dy + dp
					s = math.Max(math.Abs(theta), math.Max(math.Abs(*dy), math.Abs(dp)))
					_s := theta / s
					gamma = s * math.Sqrt((_s * _s) - *dy / s * (dp / s))
					if *stp > *sty {
						gamma = -gamma
					}
					p = gamma - dp + theta
					q = gamma - dp + gamma + *dy
					r = p / q
					stpc = *stp + r * (*sty - *stp)
					stpf = stpc
				}else {
					if *stp > *stx {
						stpf = stmax
					}else {
						stpf = stmin
					}
				}
			}
		}
	}

	//
	//     UPDATE THE INTERVAL OF UNCERTAINTY. THIS UPDATE DOES NOT
	//     DEPEND ON THE NEW STEP OR THE CASE ANALYSIS ABOVE.
	//
	if fp > *fx {
		*sty = *stp
		*fy = fp
		*dy = dp
	}else {
		if sgnd < 0.0 {
			*sty = *stx
			*fy = *fx
			*dy = *dx
		}
		*stx = *stp
		*fx = fp
		*dx = dp
	}

	//
	//     COMPUTE THE NEW STEP AND SAFEGUARD IT.
	//
	stpf = math.Min(stmax, stpf)
	stpf = math.Max(stmin, stpf)
	*stp = stpf
	if *brackt && bound {
		if *sty > *stx {
			*stp = math.Min(*stx + 0.66 * (*sty - *stx), *stp)
		}else {
			*stp = math.Max(*stx + 0.66 * (*sty - *stx), *stp)
		}
	}
}

/*************************************************************************
THE  PURPOSE  OF  MCSRCH  IS  TO  FIND A STEP WHICH SATISFIES A SUFFICIENT
DECREASE CONDITION AND A CURVATURE CONDITION.

AT EACH STAGE THE SUBROUTINE  UPDATES  AN  INTERVAL  OF  UNCERTAINTY  WITH
ENDPOINTS  STX  AND  STY.  THE INTERVAL OF UNCERTAINTY IS INITIALLY CHOSEN
SO THAT IT CONTAINS A MINIMIZER OF THE MODIFIED FUNCTION

	F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S).

IF  A STEP  IS OBTAINED FOR  WHICH THE MODIFIED FUNCTION HAS A NONPOSITIVE
FUNCTION  VALUE  AND  NONNEGATIVE  DERIVATIVE,   THEN   THE   INTERVAL  OF
UNCERTAINTY IS CHOSEN SO THAT IT CONTAINS A MINIMIZER OF F(X+STP*S).

THE  ALGORITHM  IS  DESIGNED TO FIND A STEP WHICH SATISFIES THE SUFFICIENT
DECREASE CONDITION

	F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S),

AND THE CURVATURE CONDITION

	ABS(GRADF(X+STP*S)'S)) .LE. GTOL*ABS(GRADF(X)'S).

IF  FTOL  IS  LESS  THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION IS BOUNDED
BELOW,  THEN  THERE  IS  ALWAYS  A  STEP  WHICH SATISFIES BOTH CONDITIONS.
IF  NO  STEP  CAN BE FOUND  WHICH  SATISFIES  BOTH  CONDITIONS,  THEN  THE
ALGORITHM  USUALLY STOPS  WHEN  ROUNDING ERRORS  PREVENT FURTHER PROGRESS.
IN THIS CASE STP ONLY SATISFIES THE SUFFICIENT DECREASE CONDITION.


:::::::::::::IMPORTANT NOTES:::::::::::::

NOTE 1:

This routine  guarantees that it will stop at the last point where function
value was calculated. It won't make several additional function evaluations
after finding good point. So if you store function evaluations requested by
this routine, you can be sure that last one is the point where we've stopped.

NOTE 2:

when 0<StpMax<StpMin, algorithm will terminate with INFO=5 and Stp=0.0
:::::::::::::::::::::::::::::::::::::::::


PARAMETERS DESCRIPRION

STAGE IS ZERO ON FIRST CALL, ZERO ON FINAL EXIT

N IS A POSITIVE INTEGER INPUT VARIABLE SET TO THE NUMBER OF VARIABLES.

X IS  AN  ARRAY  OF  LENGTH N. ON INPUT IT MUST CONTAIN THE BASE POINT FOR
THE LINE SEARCH. ON OUTPUT IT CONTAINS X+STP*S.

F IS  A  VARIABLE. ON INPUT IT MUST CONTAIN THE VALUE OF F AT X. ON OUTPUT
IT CONTAINS THE VALUE OF F AT X + STP*S.

G IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE GRADIENT OF F AT X.
ON OUTPUT IT CONTAINS THE GRADIENT OF F AT X + STP*S.

S IS AN INPUT ARRAY OF LENGTH N WHICH SPECIFIES THE SEARCH DIRECTION.

STP  IS  A NONNEGATIVE VARIABLE. ON INPUT STP CONTAINS AN INITIAL ESTIMATE
OF A SATISFACTORY STEP. ON OUTPUT STP CONTAINS THE FINAL ESTIMATE.

FTOL AND GTOL ARE NONNEGATIVE INPUT VARIABLES. TERMINATION OCCURS WHEN THE
SUFFICIENT DECREASE CONDITION AND THE DIRECTIONAL DERIVATIVE CONDITION ARE
SATISFIED.

XTOL IS A NONNEGATIVE INPUT VARIABLE. TERMINATION OCCURS WHEN THE RELATIVE
WIDTH OF THE INTERVAL OF UNCERTAINTY IS AT MOST XTOL.

STPMIN AND STPMAX ARE NONNEGATIVE INPUT VARIABLES WHICH SPECIFY LOWER  AND
UPPER BOUNDS FOR THE STEP.

MAXFEV IS A POSITIVE INTEGER INPUT VARIABLE. TERMINATION OCCURS WHEN THE
NUMBER OF CALLS TO FCN IS AT LEAST MAXFEV BY THE END OF AN ITERATION.

INFO IS AN INTEGER OUTPUT VARIABLE SET AS FOLLOWS:
	INFO = 0  IMPROPER INPUT PARAMETERS.

	INFO = 1  THE SUFFICIENT DECREASE CONDITION AND THE
			  DIRECTIONAL DERIVATIVE CONDITION HOLD.

	INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY
			  IS AT MOST XTOL.

	INFO = 3  NUMBER OF CALLS TO FCN HAS REACHED MAXFEV.

	INFO = 4  THE STEP IS AT THE LOWER BOUND STPMIN.

	INFO = 5  THE STEP IS AT THE UPPER BOUND STPMAX.

	INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS.
			  THERE MAY NOT BE A STEP WHICH SATISFIES THE
			  SUFFICIENT DECREASE AND CURVATURE CONDITIONS.
			  TOLERANCES MAY BE TOO SMALL.

NFEV IS AN INTEGER OUTPUT VARIABLE SET TO THE NUMBER OF CALLS TO FCN.

WA IS A WORK ARRAY OF LENGTH N.

ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. JUNE 1983
JORGE J. MORE', DAVID J. THUENTE
*************************************************************************/
func mcsrch(n int, x *[]float64, f *float64, g, s *[]float64, stp *float64, stpmax, gtol float64, info, nfev *int, wa *[]float64, state *linminstate, stage *int) {
	v := 0.0
	p5 := 0.0
	p66 := 0.0
	zero := 0.0
	i_ := 0

	//
	// init
	//
	p5 = 0.5
	p66 = 0.66
	state.xtrapf = 4.0
	zero = 0
	if stpmax == 0 {
		stpmax = defstpmax
	}
	if *stp < stpmin {
		*stp = stpmin;
	}
	if *stp > stpmax {
		*stp = stpmax
	}

	//
	// Main cycle
	//
	for {
		if *stage == 0 {
			//
			// NEXT
			//
			*stage = 2
			continue
		}
		if *stage == 2 {
			state.infoc = 1
			*info = 0

			//
			//     CHECK THE INPUT PARAMETERS FOR ERRORS.
			//
			if stpmax < stpmin && stpmax > 0 {
				*info = 5
				*stp = 0.0
				return
			}
			if ((((((n <= 0 || *stp <= 0) || ftol < 0) || gtol < zero) || xtol < zero) || stpmin < zero) || stpmax < stpmin) || maxfev <= 0 {
				*stage = 0
				return
			}

			//
			//     COMPUTE THE INITIAL GRADIENT IN THE SEARCH DIRECTION
			//     AND CHECK THAT S IS A DESCENT DIRECTION.
			//
			v = 0.0
			for i_ = 0; i_ <= n - 1; i_++ {
				v += (*g)[i_] * (*s)[i_]
			}
			state.dginit = v
			if state.dginit >= 0 {
				*stage = 0
				return
			}

			//
			//     INITIALIZE LOCAL VARIABLES.
			//
			state.brackt = false
			state.stage1 = true
			*nfev = 0
			state.finit = *f
			state.dgtest = ftol * state.dginit
			state.width = stpmax - stpmin
			state.width1 = state.width / p5
			for i_ = 0; i_ <= n - 1; i_++ {
				(*wa)[i_] = (*x)[i_]
			}

			//
			//     THE VARIABLES STX, FX, DGX CONTAIN THE VALUES OF THE STEP,
			//     FUNCTION, AND DIRECTIONAL DERIVATIVE AT THE BEST STEP.
			//     THE VARIABLES STY, FY, DGY CONTAIN THE VALUE OF THE STEP,
			//     FUNCTION, AND DERIVATIVE AT THE OTHER ENDPOINT OF
			//     THE INTERVAL OF UNCERTAINTY.
			//     THE VARIABLES STP, F, DG CONTAIN THE VALUES OF THE STEP,
			//     FUNCTION, AND DERIVATIVE AT THE CURRENT STEP.
			//
			state.stx = 0
			state.fx = state.finit
			state.dgx = state.dginit
			state.sty = 0
			state.fy = state.finit
			state.dgy = state.dginit

			//
			// NEXT
			//
			*stage = 3
			continue
		}
		if *stage == 3 {
			//
			//     START OF ITERATION.
			//
			//     SET THE MINIMUM AND MAXIMUM STEPS TO CORRESPOND
			//     TO THE PRESENT INTERVAL OF UNCERTAINTY.
			//
			if state.brackt {
				if state.stx < state.sty {
					state.stmin = state.stx
					state.stmax = state.sty
				}else {
					state.stmin = state.sty
					state.stmax = state.stx
				}
			}else {
				state.stmin = state.stx
				state.stmax = *stp + state.xtrapf * (*stp - state.stx)
			}

			//
			//        FORCE THE STEP TO BE WITHIN THE BOUNDS STPMAX AND STPMIN.
			//
			if *stp > stpmax {
				*stp = stpmax
			}
			if *stp < stpmin {
				*stp = stpmin
			}

			//
			//        IF AN UNUSUAL TERMINATION IS TO OCCUR THEN LET
			//        STP BE THE LOWEST POINT OBTAINED SO FAR.
			//

			//			(((state.brackt & ((stp)<=(state.stmin) | (stp)>=(state.stmax))) | nfev>=maxfev-1) | state.infoc==0)
			//			|
			//			(state.brackt & (state.stmax-state.stmin)<=(xtol*state.stmax))
			if (((state.brackt && (*stp <= state.stmin || *stp >= state.stmax)) || *nfev >= maxfev - 1) || state.infoc == 0) || (state.brackt && (state.stmax - state.stmin) <= (xtol * state.stmax)) {
				*stp = state.stx
			}

			//
			//        EVALUATE THE FUNCTION AND GRADIENT AT STP
			//        AND COMPUTE THE DIRECTIONAL DERIVATIVE.
			//
			for i_ = 0; i_ <= n - 1; i_++ {
				(*x)[i_] = (*wa)[i_]
			}
			for i_ = 0; i_ <= n - 1; i_++ {
				(*x)[i_] = (*x)[i_] + *stp * (*s)[i_]
			}

			//
			// NEXT
			//
			*stage = 4
			return
		}
		if *stage == 4 {
			*info = 0
			*nfev += 1
			v = 0.0
			for i_ = 0; i_ <= n - 1; i_++ {
				v += (*g)[i_] * (*s)[i_]
			}
			state.dg = v
			state.ftest1 = state.finit + *stp * state.dgtest

			//
			//        TEST FOR CONVERGENCE.
			//
			if (state.brackt && (*stp <= state.stmin || *stp >= state.stmax)) || state.infoc == 0 {
				*info = 6
			}
			if (*stp == stpmax && *f <= state.ftest1) && state.dg <= state.dgtest {
				*info = 5
			}
			if *stp == stpmin && (*f > state.ftest1 || state.dg >= state.dgtest) {
				*info = 4
			}
			if *nfev >= maxfev {
				*info = 3
			}
			if state.brackt && (state.stmax - state.stmin) <= (xtol * state.stmax) {
				*info = 2
			}
			if *f <= state.ftest1 && math.Abs(state.dg) <= -(gtol * state.dginit) {
				*info = 1
			}
			//
			//        CHECK FOR TERMINATION.
			//
			if *info != 0 {
				*stage = 0
				return
			}

			//
			//        IN THE FIRST STAGE WE SEEK A STEP FOR WHICH THE MODIFIED
			//        FUNCTION HAS A NONPOSITIVE VALUE AND NONNEGATIVE DERIVATIVE.
			//
			if (state.stage1 && *f <= state.ftest1) && state.dg >= (math.Min(ftol, gtol) * state.dginit) {
				state.stage1 = false
			}

			//
			//        A MODIFIED FUNCTION IS USED TO PREDICT THE STEP ONLY IF
			//        WE HAVE NOT OBTAINED A STEP FOR WHICH THE MODIFIED
			//        FUNCTION HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE
			//        DERIVATIVE, AND IF A LOWER FUNCTION VALUE HAS BEEN
			//        OBTAINED BUT THE DECREASE IS NOT SUFFICIENT.
			//
			if (state.stage1 && *f <= state.fx) && *f > state.ftest1 {
				//
				//           DEFINE THE MODIFIED FUNCTION AND DERIVATIVE VALUES.
				//
				state.fm = *f - *stp * state.dgtest
				state.fxm = state.fx - state.stx * state.dgtest
				state.fym = state.fy - state.sty * state.dgtest
				state.dgm = state.dg - state.dgtest
				state.dgxm = state.dgx - state.dgtest
				state.dgym = state.dgy - state.dgtest

				//
				//           CALL CSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY
				//           AND TO COMPUTE THE NEW STEP.
				//
				mcstep(&state.stx, &state.fxm, &state.dgxm, &state.sty, &state.fym, &state.dgym, stp, state.fm, state.dgm, &state.brackt, state.stmin, state.stmax, &state.infoc)

				//
				//           RESET THE FUNCTION AND GRADIENT VALUES FOR F.
				//
				state.fx = state.fxm + state.stx * state.dgtest
				state.fy = state.fym + state.sty * state.dgtest
				state.dgx = state.dgxm + state.dgtest
				state.dgy = state.dgym + state.dgtest
			}else {
				//
				//           CALL MCSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY
				//           AND TO COMPUTE THE NEW STEP.
				//
				mcstep(&state.stx, &state.fx, &state.dgx, &state.sty, &state.fy, &state.dgy, stp, *f, state.dg, &state.brackt, state.stmin, state.stmax, &state.infoc)
			}

			//
			//        FORCE A SUFFICIENT DECREASE IN THE SIZE OF THE
			//        INTERVAL OF UNCERTAINTY.
			//
			if state.brackt {
				if math.Abs(state.sty - state.stx) >= (p66 * state.width1) {
					*stp = state.stx + p5 * (state.sty - state.stx)
				}
				state.width1 = state.width
				state.width = math.Abs(state.sty - state.stx)
			}

			//
			//  NEXT.
			//
			*stage = 3
			continue
		}
	}
}

/*************************************************************************
This subroutine is used to "trim" target function, i.e. to do following
transformation:

				   { {F,G}          if F<Threshold
	{F_tr, G_tr} = {
				   { {Threshold, 0} if F>=Threshold

Such transformation allows us to  solve  problems  with  singularities  by
redefining function in such way that it becomes bounded from above.

  -- ALGLIB --
	 Copyright 10.05.2011 by Bochkanov Sergey
*************************************************************************/
func trimfunction(f *float64, g *[]float64, n int, threshold float64) {
	if *f >= threshold {
		*f = threshold
		for i := 0; i <= n - 1; i++ {
			(*g)[i] = 0.0
		}
	}
}

/*************************************************************************
NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied gradient,  and one which uses function value
   only  and  numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   (either MinLBFGSCreate() for analytical gradient  or  MinLBFGSCreateF()
   for numerical differentiation) you should choose appropriate variant of
   MinLBFGSOptimize() - one  which  accepts  function  AND gradient or one
   which accepts function ONLY.

   Be careful to choose variant of MinLBFGSOptimize() which corresponds to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to MinLBFGSOptimize()  and specific
   function used to create optimizer.


					 |         USER PASSED TO MinLBFGSOptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   MinLBFGSCreateF() |     work                FAIL
   MinLBFGSCreate()  |     FAIL                work

   Here "FAIL" denotes inappropriate combinations  of  optimizer  creation
   function  and  MinLBFGSOptimize()  version.   Attemps   to   use   such
   combination (for example, to create optimizer with MinLBFGSCreateF() and
   to pass gradient information to MinCGOptimize()) will lead to exception
   being thrown. Either  you  did  not pass gradient when it WAS needed or
   you passed gradient when it was NOT needed.

  -- ALGLIB --
	 Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
func minlbfgsiteration(state *minlbfgsstate) bool {
	n := 0
	m := 0
	i := 0
	j := 0
	ic := 0
	mcinfo := 0
	v := 0.0
	vv := 0.0
	i_ := 0


	//
	// Reverse communication preparations
	// I know it looks ugly, but it works the same way
	// anywhere from C++ to Python.
	//
	// This code initializes locals by:
	// * random values determined during code
	//   generation - on first subroutine call
	// * values from previous call - on subsequent calls
	//
	if state.rstate.stage >= 0 {
		n = state.rstate.ia[0]
		m = state.rstate.ia[1]
		i = state.rstate.ia[2]
		j = state.rstate.ia[3]
		ic = state.rstate.ia[4]
		mcinfo = state.rstate.ia[5]
		v = state.rstate.ra[0]
		vv = state.rstate.ra[1]
	}else {
		n = -983
		m = -989
		i = -834
		j = 900
		ic = -287
		mcinfo = 364
		v = 214
		vv = -338
	}
	if state.rstate.stage == 0 {
		goto lbl_0
	}
	if state.rstate.stage == 1 {
		goto lbl_1
	}
	if state.rstate.stage == 2 {
		goto lbl_2
	}
	if state.rstate.stage == 3 {
		goto lbl_3
	}
	if state.rstate.stage == 4 {
		goto lbl_4
	}
	if state.rstate.stage == 5 {
		goto lbl_5
	}
	if state.rstate.stage == 6 {
		goto lbl_6
	}
	if state.rstate.stage == 7 {
		goto lbl_7
	}
	if state.rstate.stage == 8 {
		goto lbl_8
	}
	if state.rstate.stage == 9 {
		goto lbl_9
	}
	if state.rstate.stage == 10 {
		goto lbl_10
	}
	if state.rstate.stage == 11 {
		goto lbl_11
	}
	if state.rstate.stage == 12 {
		goto lbl_12
	}
	if state.rstate.stage == 13 {
		goto lbl_13
	}

	//
	// Routine body
	//

	//
	// Unload frequently used variables from State structure
	// (just for typing convinience)
	//
	n = state.n
	m = state.m
	state.repterminationtype = 0
	state.repiterationscount = 0
	state.repnfev = 0

	//
	// Calculate F/G at the initial point
	//
	clearrequestfields(state)
	if state.diffstep != 0 {
		goto lbl_14
	}
	state.needfg = true
	state.rstate.stage = 0
	goto lbl_rcomm
	lbl_0:
	state.needfg = false
	goto lbl_15
	lbl_14:
	state.needf = true
	state.rstate.stage = 1
	goto lbl_rcomm
	lbl_1:
	state.fbase = state.f
	i = 0
	lbl_16:
	if i > n - 1 {
		goto lbl_18
	}
	v = state.x[i]
	state.x[i] = v - state.diffstep * state.s[i]
	state.rstate.stage = 2
	goto lbl_rcomm
	lbl_2:
	state.fm2 = state.f
	state.x[i] = v - 0.5 * state.diffstep * state.s[i]
	state.rstate.stage = 3
	goto lbl_rcomm
	lbl_3:
	state.fm1 = state.f
	state.x[i] = v + 0.5 * state.diffstep * state.s[i]
	state.rstate.stage = 4
	goto lbl_rcomm
	lbl_4:
	state.fp1 = state.f
	state.x[i] = v + state.diffstep * state.s[i]
	state.rstate.stage = 5
	goto lbl_rcomm
	lbl_5:
	state.fp2 = state.f
	state.x[i] = v
	state.g[i] = (8 * (state.fp1 - state.fm1) - (state.fp2 - state.fm2)) / (6 * state.diffstep * state.s[i])
	i = i + 1
	goto lbl_16
	lbl_18:
	state.f = state.fbase
	state.needf = false
	lbl_15:
	trimprepare(state.f, &state.trimthreshold)
	if !state.xrep {
		goto lbl_19
	}
	clearrequestfields(state)
	state.xupdated = true
	state.rstate.stage = 6
	goto lbl_rcomm
	lbl_6:
	state.xupdated = false
	lbl_19:
	state.repnfev = 1
	state.fold = state.f
	v = 0
	for i = 0; i <= n - 1; i++ {
		_v2 := state.g[i] * state.s[i]
		v = v + (_v2 * _v2)
	}
	if math.Sqrt(v) <= float64(state.epsg) {
		state.repterminationtype = 4
		return false
	}

	//
	// Choose initial step and direction.
	// Apply preconditioner, if we have something other than default.
	//
	for i_ = 0; i_ <= n - 1; i_++ {
		state.d[i_] = -state.g[i_]
	}
	if state.prectype == 0 {
		//
		// Default preconditioner is used, but we can't use it before iterations will start
		//
		v = 0.0
		for i_ = 0; i_ <= n - 1; i_++ {
			v += state.g[i_] * state.g[i_]
		}
		v = math.Sqrt(v)
		if state.stpmax == 0 {
			state.stp = math.Min(1.0 / v, 1)
		}else {
			state.stp = math.Min(1.0 / v, state.stpmax)
		}
	}
	if state.prectype == 1 {
		//
		// Cholesky preconditioner is used
		//
		fblscholeskysolve(&state.denseh, 1.0, n, true, &state.d, &state.autobuf)
		state.stp = 1
	}
	if state.prectype == 2 {
		//
		// diagonal approximation is used
		//
		for i = 0; i <= n - 1; i++ {
			state.d[i] = state.d[i] / state.diagh[i]
		}
		state.stp = 1
	}
	if state.prectype == 3 {
		//
		// scale-based preconditioner is used
		//
		for i = 0; i <= n - 1; i++ {
			state.d[i] = state.d[i] * state.s[i] * state.s[i]
		}
		state.stp = 1
	}

	//
	// Main cycle
	//
	state.k = 0
	lbl_21:
	if false {
		goto lbl_22
	}

	//
	// Main cycle: prepare to 1-D line search
	//
	state.p = state.k % m
	state.q = utils.MinInt(state.k, m - 1)

	//
	// Store X[k], G[k]
	//
	for i_ = 0; i_ <= n - 1; i_++ {
		state.sk[state.p][ i_] = -state.x[i_]
	}
	for i_ = 0; i_ <= n - 1; i_++ {
		state.yk[state.p][i_] = -state.g[i_]
	}

	//
	// Minimize F(x+alpha*d)
	// Calculate S[k], Y[k]
	//
	state.mcstage = 0
	if state.k != 0 {
		state.stp = 1.0
	}
	linminnormalized(&state.d, &state.stp, n)
	mcsrch(n, &state.x, &state.f, &state.g, &state.d, &state.stp, state.stpmax, gtol, &mcinfo, &state.nfev, &state.work, state.lstate, &state.mcstage)
	lbl_23:
	if state.mcstage == 0 {
		goto lbl_24
	}
	clearrequestfields(state)
	if state.diffstep != 0 {
		goto lbl_25
	}
	state.needfg = true
	state.rstate.stage = 7
	goto lbl_rcomm
	lbl_7:
	state.needfg = false
	goto lbl_26
	lbl_25:
	state.needf = true
	state.rstate.stage = 8
	goto lbl_rcomm
	lbl_8:
	state.fbase = state.f
	i = 0
	lbl_27:
	if i > n - 1 {
		goto lbl_29
	}
	v = state.x[i]
	state.x[i] = v - state.diffstep * state.s[i]
	state.rstate.stage = 9
	goto lbl_rcomm
	lbl_9:
	state.fm2 = state.f
	state.x[i] = v - 0.5 * state.diffstep * state.s[i]
	state.rstate.stage = 10
	goto lbl_rcomm
	lbl_10:
	state.fm1 = state.f
	state.x[i] = v + 0.5 * state.diffstep * state.s[i]
	state.rstate.stage = 11
	goto lbl_rcomm
	lbl_11:
	state.fp1 = state.f
	state.x[i] = v + state.diffstep * state.s[i]
	state.rstate.stage = 12
	goto lbl_rcomm
	lbl_12:
	state.fp2 = state.f
	state.x[i] = v
	state.g[i] = (8 * (state.fp1 - state.fm1) - (state.fp2 - state.fm2)) / (6 * state.diffstep * state.s[i])
	i = i + 1
	goto lbl_27
	lbl_29:
	state.f = state.fbase
	state.needf = false
	lbl_26:
	trimfunction(&state.f, &state.g, n, state.trimthreshold)
	mcsrch(n, &state.x, &state.f, &state.g, &state.d, &state.stp, state.stpmax, gtol, &mcinfo, &state.nfev, &state.work, state.lstate, &state.mcstage)
	goto lbl_23
	lbl_24:
	if !state.xrep {
		goto lbl_30
	}

	//
	// report
	//
	clearrequestfields(state)
	state.xupdated = true
	state.rstate.stage = 13
	goto lbl_rcomm
	lbl_13:
	state.xupdated = false
	lbl_30:
	state.repnfev = state.repnfev + state.nfev
	state.repiterationscount = state.repiterationscount + 1
	for i_ = 0; i_ <= n - 1; i_++ {
		state.sk[state.p][ i_] = state.sk[state.p][ i_] + state.x[i_]
	}
	for i_ = 0; i_ <= n - 1; i_++ {
		state.yk[state.p][ i_] = state.yk[state.p][ i_] + state.g[i_]
	}

	//
	// Stopping conditions
	//
	if state.repiterationscount >= state.maxits && state.maxits > 0 {
		//
		// Too many iterations
		//
		state.repterminationtype = 5
		return false
	}
	v = 0
	for i = 0; i <= n - 1; i++ {
		_v2 := state.g[i] * state.s[i]
		v = v + (_v2 * _v2)
	}
	if math.Sqrt(v) <= float64(state.epsg) {
		//
		// Gradient is small enough
		//
		state.repterminationtype = 4
		return false
	}
	if (state.fold - state.f) <= (state.epsf * math.Max(math.Abs(state.fold), math.Max(math.Abs(state.f), 1.0))) {
		//
		// F(k+1)-F(k) is small enough
		//
		state.repterminationtype = 1
		return false
	}
	v = 0
	for i = 0; i <= n - 1; i++ {
		_v2 := state.sk[state.p][ i] / state.s[i]
		v = v + (_v2 * _v2)
	}
	if math.Sqrt(v) <= float64(state.epsx) {
		//
		// X(k+1)-X(k) is small enough
		//
		state.repterminationtype = 2
		return false
	}

	//
	// If Wolfe conditions are satisfied, we can update
	// limited memory model.
	//
	// However, if conditions are not satisfied (NFEV limit is met,
	// function is too wild, ...), we'll skip L-BFGS update
	//
	if mcinfo != 1 {
		//
		// Skip update.
		//
		// In such cases we'll initialize search direction by
		// antigradient vector, because it  leads to more
		// transparent code with less number of special cases
		//
		state.fold = state.f
		for i_ = 0; i_ <= n - 1; i_++ {
			state.d[i_] = -state.g[i_]
		}
	}else {
		//
		// Calculate Rho[k], GammaK
		//
		v = 0.0
		for i_ = 0; i_ <= n - 1; i_++ {
			v += state.yk[state.p][i_] * state.sk[state.p][ i_]
		}
		vv = 0.0
		for i_ = 0; i_ <= n - 1; i_++ {
			vv += state.yk[state.p][i_] * state.yk[state.p][ i_]
		}
		if v == 0 || vv == 0 {
			//
			// Rounding errors make further iterations impossible.
			//
			state.repterminationtype = -2
			return false
		}
		state.rho[state.p] = 1 / v
		state.gammak = v / vv

		//
		//  Calculate d(k+1) = -H(k+1)*g(k+1)
		//
		//  for I:=K downto K-Q do
		//      V = s(i)^T * work(iteration:I)
		//      theta(i) = V
		//      work(iteration:I+1) = work(iteration:I) - V*Rho(i)*y(i)
		//  work(last iteration) = H0*work(last iteration) - preconditioner
		//  for I:=K-Q to K do
		//      V = y(i)^T*work(iteration:I)
		//      work(iteration:I+1) = work(iteration:I) +(-V+theta(i))*Rho(i)*s(i)
		//
		//  NOW WORK CONTAINS d(k+1)
		//
		for i_ = 0; i_ <= n - 1; i_++ {
			state.work[i_] = state.g[i_]
		}
		for i = state.k; i >= state.k - state.q; i-- {
			ic = i % m
			v = 0.0
			for i_ = 0; i_ <= n - 1; i_++ {
				v += state.sk[ic][ i_] * state.work[i_]
			}
			state.theta[ic] = v
			vv = v * state.rho[ic]
			for i_ = 0; i_ <= n - 1; i_++ {
				state.work[i_] = state.work[i_] - vv * state.yk[ic][ i_]
			}
		}
		if state.prectype == 0 {
			//
			// Simple preconditioner is used
			//
			v = state.gammak
			for i_ = 0; i_ <= n - 1; i_++ {
				state.work[i_] = v * state.work[i_]
			}
		}
		if state.prectype == 1 {
			//
			// Cholesky preconditioner is used
			//
			fblscholeskysolve(&state.denseh, 1, n, true, &state.work, &state.autobuf)
		}
		if state.prectype == 2 {
			//
			// diagonal approximation is used
			//
			for i = 0; i <= n - 1; i++ {
				state.work[i] = state.work[i] / state.diagh[i]
			}
		}
		if state.prectype == 3 {
			//
			// scale-based preconditioner is used
			//
			for i = 0; i <= n - 1; i++ {
				state.work[i] = state.work[i] * state.s[i] * state.s[i]
			}
		}
		for i = state.k - state.q; i <= state.k; i++ {
			ic = i % m
			v = 0.0
			for i_ = 0; i_ <= n - 1; i_++ {
				v += state.yk[ic][i_] * state.work[i_]
			}
			vv = state.rho[ic] * (-v + state.theta[ic])
			for i_ = 0; i_ <= n - 1; i_++ {
				state.work[i_] = state.work[i_] + vv * state.sk[ic][ i_]
			}
		}
		for i_ = 0; i_ <= n - 1; i_++ {
			state.d[i_] = -state.work[i_]
		}

		//
		// Next step
		//
		state.fold = state.f
		state.k += 1
	}
	goto lbl_21
	lbl_22:
	return false;

	//
	// Saving state
	//
	lbl_rcomm:
	state.rstate.ia[0] = n
	state.rstate.ia[1] = m
	state.rstate.ia[2] = i
	state.rstate.ia[3] = j
	state.rstate.ia[4] = ic
	state.rstate.ia[5] = mcinfo
	state.rstate.ra[0] = v
	state.rstate.ra[1] = vv

	return true
}

/*************************************************************************
L-BFGS algorithm results

INPUT PARAMETERS:
	State   -   algorithm state

OUTPUT PARAMETERS:
	X       -   array[0..N-1], solution
	Rep     -   optimization report:
				* Rep.TerminationType completetion code:
					* -2    rounding errors prevent further improvement.
							X contains best point found.
					* -1    incorrect parameters were specified
					*  1    relative function improvement is no more than
							EpsF.
					*  2    relative step is no more than EpsX.
					*  4    gradient norm is no more than EpsG
					*  5    MaxIts steps was taken
					*  7    stopping conditions are too stringent,
							further improvement is impossible
				* Rep.IterationsCount contains iterations count
				* NFEV countains number of function calculations

  -- ALGLIB --
	 Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgsresults(state *minlbfgsstate, x *[]float64, rep *minlbfgsreport) {
	*x = make([]float64, 0)
	minlbfgsresultsbuf(state, x, rep)
}

/*************************************************************************
L-BFGS algorithm results

Buffered implementation of MinLBFGSResults which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
	 Copyright 20.08.2010 by Bochkanov Sergey
*************************************************************************/
func minlbfgsresultsbuf(state *minlbfgsstate, x *[]float64, rep *minlbfgsreport) {
	i_ := 0

	if len(*x) < state.n {
		*x = make([]float64, state.n)
	}
	for i_ = 0; i_ <= state.n - 1; i_++ {
		(*x)[i_] = state.x[i_]
	}
	rep.iterationscount = state.repiterationscount
	rep.nfev = state.repnfev
	rep.terminationtype = state.repterminationtype
}

/*************************************************************************
Returns block size - subdivision size where  cache-oblivious  soubroutines
switch to the optimized kernel.

INPUT PARAMETERS
	A   -   real matrix, is passed to ensure that we didn't split
			complex matrix using real splitting subroutine.
			matrix itself is not changed.

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func ablasblocksize(a *[][]float64) int {
	return 32
}

/*************************************************************************
Fast kernel

  -- ALGLIB routine --
	 19.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixmvf(m, n int, a *[][]float64, ia, ja, opa int, x *[]float64, ix int, y *[]float64, iy int) bool {
	return false
}

/*************************************************************************
Matrix-vector product: y := op(A)*x

INPUT PARAMETERS:
	M   -   number of rows of op(A)
	N   -   number of columns of op(A)
	A   -   target matrix
	IA  -   submatrix offset (row index)
	JA  -   submatrix offset (column index)
	OpA -   operation type:
			* OpA=0     =>  op(A) = A
			* OpA=1     =>  op(A) = A^T
	X   -   input vector
	IX  -   subvector offset
	IY  -   subvector offset

OUTPUT PARAMETERS:
	Y   -   vector which stores result

if M=0, then subroutine does nothing.
if N=0, Y is filled by zeros.


  -- ALGLIB routine --

	 28.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixmv(m, n int, a *[][]float64, ia, ja, opa int, x *[]float64, ix int, y *[]float64, iy int) {
	i := 0
	v := 0.0
	i_ := 0
	i1_ := 0

	if m == 0 {
		return
	}
	if n == 0 {
		for i = 0; i <= m - 1; i++ {
			(*y)[iy + i] = 0
		}
		return
	}
	if rmatrixmvf(m, n, a, ia, ja, opa, x, ix, y, iy) {
		return
	}
	if opa == 0 {
		//
		// y = A*x
		//
		for i = 0; i <= m - 1; i++ {
			i1_ = (ix) - (ja)
			v = 0.0
			for i_ = ja; i_ <= ja + n - 1; i_++ {
				v += (*a)[ia + i][ i_] * (*x)[i_ + i1_]
			}
			(*y)[iy + i] = v
		}
		return
	}
	if opa == 1 {
		//
		// y = A^T*x
		//
		for i = 0; i <= m - 1; i++ {
			(*y)[iy + i] = 0
		}
		for i = 0; i <= n - 1; i++ {
			v = (*x)[ix + i]
			i1_ = (ja) - (iy)
			for i_ = iy; i_ <= iy + m - 1; i_++ {
				(*y)[i_] = (*y)[i_] + v * (*a)[ia + i][ i_ + i1_]
			}
		}
		return
	}
}

/*************************************************************************
Level-2 Cholesky subroutine

  -- LAPACK routine (version 3.0) --
	 Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
	 Courant Institute, Argonne National Lab, and Rice University
	 February 29, 1992
*************************************************************************/
func spdmatrixcholesky2(aaa *[][]float64, offs, n int, isupper bool, tmp *[]float64) bool {
	i := 0
	j := 0
	ajj := 0.0
	v := 0.0
	r := 0.0
	i_ := 0
	i1_ := 0

	result := true
	if n < 0 {
		return false
	}

	//
	// Quick return if possible
	//
	if n == 0 {
		return result
	}
	if isupper {
		//
		// Compute the Cholesky factorization A = U'*U.
		//
		for j = 0; j <= n - 1; j++ {
			//
			// Compute U(J,J) and test for non-positive-definiteness.
			//
			v = 0.0
			for i_ = offs; i_ <= offs + j - 1; i_++ {
				v += (*aaa)[i_][ offs + j] * (*aaa)[i_][ offs + j]
			}
			ajj = (*aaa)[offs + j][ offs + j] - v
			if ajj <= 0 {
				(*aaa)[offs + j][ offs + j] = ajj
				return false
			}
			ajj = math.Sqrt(ajj)
			(*aaa)[offs + j][ offs + j] = ajj

			//
			// Compute elements J+1:N-1 of row J.
			//
			if j < n - 1 {
				if j > 0 {
					i1_ = (offs) - (0)
					for i_ = 0; i_ <= j - 1; i_++ {
						(*tmp)[i_] = -(*aaa)[i_ + i1_][ offs + j]
					}
					rmatrixmv(n - j - 1, j, aaa, offs, offs + j + 1, 1, tmp, 0, tmp, n)
					i1_ = (n) - (offs + j + 1)
					for i_ = offs + j + 1; i_ <= offs + n - 1; i_++ {
						(*aaa)[offs + j][ i_] = (*aaa)[offs + j][ i_] + (*tmp)[i_ + i1_]
					}
				}
				r = 1 / ajj;
				for i_ = offs + j + 1; i_ <= offs + n - 1; i_++ {
					(*aaa)[offs + j][ i_] = r * (*aaa)[offs + j][ i_]
				}
			}
		}
	}else {
		//
		// Compute the Cholesky factorization A = L*L'.
		//
		for j = 0; j <= n - 1; j++ {
			//
			// Compute L(J+1,J+1) and test for non-positive-definiteness.
			//
			v = 0.0
			for i_ = offs; i_ <= offs + j - 1; i_++ {
				v += (*aaa)[offs + j][ i_] * (*aaa)[offs + j][ i_]
			}
			ajj = (*aaa)[offs + j][ offs + j] - v;
			if ajj <= 0 {
				(*aaa)[offs + j][ offs + j] = ajj
				result = false
				return result
			}
			ajj = math.Sqrt(ajj)
			(*aaa)[offs + j][ offs + j] = ajj

			//
			// Compute elements J+1:N of column J.
			//
			if j < n - 1 {
				if j > 0 {
					i1_ = (offs) - (0)
					for i_ = 0; i_ <= j - 1; i_++ {
						(*tmp)[i_] = (*aaa)[offs + j][ i_ + i1_]
					}
					rmatrixmv(n - j - 1, j, aaa, offs + j + 1, offs, 0, tmp, 0, tmp, n)
					for i = 0; i <= n - j - 2; i++ {
						(*aaa)[offs + j + 1 + i][offs + j] = ((*aaa)[offs + j + 1 + i][ offs + j] - (*tmp)[n + i]) / ajj
					}
				}else {
					for i = 0; i <= n - j - 2; i++ {
						(*aaa)[offs + j + 1 + i][offs + j] = (*aaa)[offs + j + 1 + i][ offs + j] / ajj
					}
				}
			}
		}
	}
	return result
}

/*************************************************************************
Microblock size

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func ablasmicroblocksize() int {
	return 8
}

/*************************************************************************
Complex ABLASSplitLength

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func ablasinternalsplitlength(n, nb int, n1, n2 *int) {
	r := 0

	*n1 = 0
	*n2 = 0

	if n <= nb {
		//
		// Block size, no further splitting
		//
		*n1 = n
		*n2 = 0
	}else {
		//
		// Greater than block size
		//
		if n % nb != 0 {
			//
			// Split remainder
			//
			*n2 = n % nb
			*n1 = n - *n2
		}else {
			//
			// Split on block boundaries
			//
			*n2 = n / 2
			*n1 = n - *n2
			if *n1 % nb == 0 {
				return
			}
			r = nb - *n1 % nb
			*n1 = *n1 + r
			*n2 = *n2 - r
		}
	}
}

/*************************************************************************
Splits matrix length in two parts, left part should match ABLAS block size

INPUT PARAMETERS
	A   -   real matrix, is passed to ensure that we didn't split
			complex matrix using real splitting subroutine.
			matrix itself is not changed.
	N   -   length, N>0

OUTPUT PARAMETERS
	N1  -   length
	N2  -   length

N1+N2=N, N1>=N2, N2 may be zero

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func ablassplitlength(a *[][]float64, n int, n1, n2 *int) {
	*n1 = 0
	*n2 = 0

	if n > ablasblocksize(a) {
		ablasinternalsplitlength(n, ablasblocksize(a), n1, n2)
	}else {
		ablasinternalsplitlength(n, ablasmicroblocksize(), n1, n2)
	}
}

/*************************************************************************
Fast kernel

  -- ALGLIB routine --
	 19.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixlefttrsmf(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) bool {
	return false
}

/*************************************************************************
Level 2 subroutine
*************************************************************************/
func rmatrixlefttrsm2(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) {
	i := 0
	j := 0
	vr := 0.0
	vd := 0.0
	i_ := 0


	//
	// Special case
	//
	if n * m == 0 {
		return
	}

	//
	// Try fast code
	//
	if rmatrixlefttrsmf(m, n, a, i1, j1, isupper, isunit, optype, x, i2, j2) {
		return
	}

	//
	// General case
	//
	if isupper {
		//
		// Upper triangular matrix
		//
		if optype == 0 {
			//
			// A^(-1)*X
			//
			for i = m - 1; i >= 0; i-- {
				for j = i + 1; j <= m - 1; j++ {
					vr = (*a)[i1 + i][ j1 + j]
					for i_ = j2; i_ <= j2 + n - 1; i_++ {
						(*x)[i2 + i][ i_] = (*x)[i2 + i][ i_] - vr * (*x)[i2 + j][ i_]
					}
				}
				if !isunit {
					vd = 1 / (*a)[i1 + i][ j1 + i]
					for i_ = j2; i_ <= j2 + n - 1; i_++ {
						(*x)[i2 + i][ i_] = vd * (*x)[i2 + i][ i_]
					}
				}
			}
			return
		}
		if optype == 1 {
			//
			// A^(-T)*X
			//
			for i = 0; i <= m - 1; i++ {
				if isunit {
					vd = 1
				}else {
					vd = 1 / (*a)[i1 + i][j1 + i]
				}
				for i_ = j2; i_ <= j2 + n - 1; i_++ {
					(*x)[i2 + i][ i_] = vd * (*x)[i2 + i][ i_]
				}
				for j = i + 1; j <= m - 1; j++ {
					vr = (*a)[i1 + i][ j1 + j]
					for i_ = j2; i_ <= j2 + n - 1; i_++ {
						(*x)[i2 + j][ i_] = (*x)[i2 + j][ i_] - vr * (*x)[i2 + i][ i_]
					}
				}
			}
			return
		}
	}else {
		//
		// Lower triangular matrix
		//
		if optype == 0 {
			//
			// A^(-1)*X
			//
			for i = 0; i <= m - 1; i++ {
				for j = 0; j <= i - 1; j++ {
					vr = (*a)[i1 + i][ j1 + j]
					for i_ = j2; i_ <= j2 + n - 1; i_++ {
						(*x)[i2 + i][ i_] = (*x)[i2 + i][ i_] - vr * (*x)[i2 + j][ i_]
					}
				}
				if isunit {
					vd = 1
				}else {
					vd = 1 / (*a)[i1 + j][j1 + j]
				}
				for i_ = j2; i_ <= j2 + n - 1; i_++ {
					(*x)[i2 + i][ i_] = vd * (*x)[i2 + i][ i_]
				}
			}
			return
		}
		if optype == 1 {
			//
			// A^(-T)*X
			//
			for i = m - 1; i >= 0; i-- {
				if isunit {
					vd = 1
				}else {
					vd = 1 / (*a)[i1 + i][j1 + i]
				}
				for i_ = j2; i_ <= j2 + n - 1; i_++ {
					(*x)[i2 + i][ i_] = vd * (*x)[i2 + i][ i_]
				}
				for j = i - 1; j >= 0; j-- {
					vr = (*a)[i1 + i][ j1 + j]
					for i_ = j2; i_ <= j2 + n - 1; i_++ {
						(*x)[i2 + j][ i_] = (*x)[i2 + j][ i_] - vr * (*x)[i2 + i][ i_]
					}
				}
			}
			return
		}
	}
}

/*************************************************************************
Fast kernel

  -- ALGLIB routine --
	 19.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixgemmf(m, n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, b *[][]float64, ib, jb, optypeb int, beta float64, c *[][]float64, ic, jc int) bool {
	return false
}

/*************************************************************************
GEMM kernel

  -- ALGLIB routine --
	 16.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixgemmk(m, n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, b *[][]float64, ib, jb, optypeb int, beta float64, c *[][]float64, ic, jc int) {
	i := 0
	j := 0
	v := 0.0
	i_ := 0
	i1_ := 0

	//
	// if matrix size is zero
	//
	if m * n == 0 {
		return
	}

	//
	// Try optimized code
	//
	if rmatrixgemmf(m, n, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc) {
		return
	}

	//
	// if K=0, then C=Beta*C
	//
	if k == 0 {
		if beta != 1 {
			if beta != 0 {
				for i = 0; i <= m - 1; i++ {
					for j = 0; j <= n - 1; j++ {
						(*c)[ic + i][ jc + j] = beta * (*c)[ic + i][ jc + j]
					}
				}
			}else {
				for i = 0; i <= m - 1; i++ {
					for j = 0; j <= n - 1; j++ {
						(*c)[ic + i][ jc + j] = 0            }
				}
			}
		}
		return
	}

	//
	// General case
	//
	if optypea == 0 && optypeb != 0 {
		//
		// A*B'
		//
		for i = 0; i <= m - 1; i++ {
			for j = 0; j <= n - 1; j++ {
				if k == 0 || alpha == 0 {
					v = 0
				}else {
					i1_ = (jb) - (ja)
					v = 0.0
					for i_ = ja; i_ <= ja + k - 1; i_++ {
						v += (*a)[ia + i][ i_] * (*b)[ib + j][ i_ + i1_]
					}
				}
				if beta == 0 {
					(*c)[ic + i][ jc + j] = alpha * v
				}else {
					(*c)[ic + i][ jc + j] = beta * (*c)[ic + i][ jc + j] + alpha * v
				}
			}
		}
		return
	}
	if optypea == 0 && optypeb == 0 {
		//
		// A*B
		//
		for i = 0; i <= m - 1; i++ {
			if beta != 0 {
				for i_ = jc; i_ <= jc + n - 1; i_++ {
					(*c)[ic + i][ i_] = beta * (*c)[ic + i][ i_]
				}
			}else {
				for j = 0; j <= n - 1; j++ {
					(*c)[ic + i][ jc + j] = 0
				}
			}
			if alpha != 0 {
				for j = 0; j <= k - 1; j++ {
					v = alpha * (*a)[ia + i][ja + j]
					i1_ = (jb) - (jc)
					for i_ = jc; i_ <= jc + n - 1; i_++ {
						(*c)[ic + i][ i_] = (*c)[ic + i][ i_] + v * (*b)[ib + j][ i_ + i1_]
					}
				}
			}
		}
		return
	}
	if optypea != 0 && optypeb != 0 {
		//
		// A'*B'
		//
		for i = 0; i <= m - 1; i++ {
			for j = 0; j <= n - 1; j++ {
				if alpha == 0 {
					v = 0
				}else {
					i1_ = (jb) - (ia)
					v = 0.0
					for i_ = ia; i_ <= ia + k - 1; i_++ {
						v += (*a)[i_][ ja + i] * (*b)[ib + j][ i_ + i1_]
					}
				}
				if beta == 0 {
					(*c)[ic + i][ jc + j] = alpha * v
				}else {
					(*c)[ic + i][ jc + j] = beta * (*c)[ic + i][ jc + j] + alpha * v
				}
			}
		}
		return
	}
	if optypea != 0 && optypeb == 0 {
		//
		// A'*B
		//
		if beta == 0 {
			for i = 0; i <= m - 1; i++ {
				for j = 0; j <= n - 1; j++ {
					(*c)[ic + i][ jc + j] = 0
				}
			}
		}else {
			for i = 0; i <= m - 1; i++ {
				for i_ = jc; i_ <= jc + n - 1; i_++ {
					(*c)[ic + i][ i_] = beta * (*c)[ic + i][ i_]
				}
			}
		}
		if alpha != 0 {
			for j = 0; j <= k - 1; j++ {
				for i = 0; i <= m - 1; i++ {
					v = alpha * (*a)[ia + j][ja + i]
					i1_ = (jb) - (jc)
					for i_ = jc; i_ <= jc + n - 1; i_++ {
						(*c)[ic + i][ i_] = (*c)[ic + i][ i_] + v * (*b)[ib + j][ i_ + i1_]
					}
				}
			}
		}
		return
	}
}

/*************************************************************************
Same as CMatrixGEMM, but for real numbers.
OpType may be only 0 or 1.

  -- ALGLIB routine --
	 16.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixgemm(m, n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, b *[][]float64, ib, jb, optypeb int, beta float64, c *[][]float64, ic, jc int) {
	s1 := 0
	s2 := 0
	bs := 0

	bs = ablasblocksize(a)
	if (m <= bs && n <= bs) && k <= bs {
		rmatrixgemmk(m, n, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
		return
	}
	if m >= n && m >= k {
		//
		// A*B = (A1 A2)^T*B
		//
		ablassplitlength(a, m, &s1, &s2)
		if optypea == 0 {
			rmatrixgemm(s1, n, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(s2, n, k, alpha, a, ia + s1, ja, optypea, b, ib, jb, optypeb, beta, c, ic + s1, jc)
		}else {
			rmatrixgemm(s1, n, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(s2, n, k, alpha, a, ia, ja + s1, optypea, b, ib, jb, optypeb, beta, c, ic + s1, jc)
		}
		return
	}
	if n >= m && n >= k {
		//
		// A*B = A*(B1 B2)
		//
		ablassplitlength(a, n, &s1, &s2)
		if optypeb == 0 {
			rmatrixgemm(m, s1, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(m, s2, k, alpha, a, ia, ja, optypea, b, ib, jb + s1, optypeb, beta, c, ic, jc + s1)
		}else {
			rmatrixgemm(m, s1, k, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc);
			rmatrixgemm(m, s2, k, alpha, a, ia, ja, optypea, b, ib + s1, jb, optypeb, beta, c, ic, jc + s1)
		}
		return
	}
	if k >= m && k >= n {
		//
		// A*B = (A1 A2)*(B1 B2)^T
		//
		ablassplitlength(a, k, &s1, &s2)
		if optypea == 0 && optypeb == 0 {
			rmatrixgemm(m, n, s1, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(m, n, s2, alpha, a, ia, ja + s1, optypea, b, ib + s1, jb, optypeb, 1.0, c, ic, jc)
		}
		if optypea == 0 && optypeb != 0 {
			rmatrixgemm(m, n, s1, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(m, n, s2, alpha, a, ia, ja + s1, optypea, b, ib, jb + s1, optypeb, 1.0, c, ic, jc)
		}
		if optypea != 0 && optypeb == 0 {
			rmatrixgemm(m, n, s1, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(m, n, s2, alpha, a, ia + s1, ja, optypea, b, ib + s1, jb, optypeb, 1.0, c, ic, jc)
		}
		if optypea != 0 && optypeb != 0 {
			rmatrixgemm(m, n, s1, alpha, a, ia, ja, optypea, b, ib, jb, optypeb, beta, c, ic, jc)
			rmatrixgemm(m, n, s2, alpha, a, ia + s1, ja, optypea, b, ib, jb + s1, optypeb, 1.0, c, ic, jc)
		}
		return
	}
}

/*************************************************************************
Same as CMatrixLeftTRSM, but for real matrices

OpType may be only 0 or 1.

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixlefttrsm(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) {
	s1 := 0
	s2 := 0
	bs := 0

	bs = ablasblocksize(a)
	if m <= bs && n <= bs {
		rmatrixlefttrsm2(m, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
		return
	}
	if n >= m {
		//
		// Split X: op(A)^-1*X = op(A)^-1*(X1 X2)
		//
		ablassplitlength(x, n, &s1, &s2)
		rmatrixlefttrsm(m, s1, a, i1, j1, isupper, isunit, optype, x, i2, j2)
		rmatrixlefttrsm(m, s2, a, i1, j1, isupper, isunit, optype, x, i2, j2 + s1)
	}else {
		//
		// Split A
		//
		ablassplitlength(a, m, &s1, &s2)
		if isupper && optype == 0 {
			//
			//           (A1  A12)-1  ( X1 )
			// A^-1*X* = (       )   *(    )
			//           (     A2)    ( X2 )
			//
			rmatrixlefttrsm(s2, n, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2 + s1, j2)
			rmatrixgemm(s1, n, s2, -1.0, a, i1, j1 + s1, 0, x, i2 + s1, j2, 0, 1.0, x, i2, j2)
			rmatrixlefttrsm(s1, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			return
		}
		if isupper && optype != 0 {
			//
			//          (A1'     )-1 ( X1 )
			// A^-1*X = (        )  *(    )
			//          (A12' A2')   ( X2 )
			//
			rmatrixlefttrsm(s1, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			rmatrixgemm(s2, n, s1, -1.0, a, i1, j1 + s1, optype, x, i2, j2, 0, 1.0, x, i2 + s1, j2)
			rmatrixlefttrsm(s2, n, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2 + s1, j2)
			return
		}
		if !isupper && optype == 0 {
			//
			//          (A1     )-1 ( X1 )
			// A^-1*X = (       )  *(    )
			//          (A21  A2)   ( X2 )
			//
			rmatrixlefttrsm(s1, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			rmatrixgemm(s2, n, s1, -1.0, a, i1 + s1, j1, 0, x, i2, j2, 0, 1.0, x, i2 + s1, j2)
			rmatrixlefttrsm(s2, n, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2 + s1, j2)
			return
		}
		if !isupper && optype != 0 {
			//
			//          (A1' A21')-1 ( X1 )
			// A^-1*X = (        )  *(    )
			//          (     A2')   ( X2 )
			//
			rmatrixlefttrsm(s2, n, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2 + s1, j2)
			rmatrixgemm(s1, n, s2, -1.0, a, i1 + s1, j1, optype, x, i2 + s1, j2, 0, 1.0, x, i2, j2)
			rmatrixlefttrsm(s1, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			return
		}
	}
}

/*************************************************************************
Fast kernel

  -- ALGLIB routine --
	 19.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixsyrkf(n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, beta float64, c *[][]float64, ic, jc int, isupper bool) bool {
	return false
}

/*************************************************************************
Level 2 subrotuine
*************************************************************************/
func rmatrixsyrk2(n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, beta float64, c *[][]float64, ic, jc int, isupper bool) {
	i := 0
	j := 0
	j1 := 0
	j2 := 0
	v := 0.0
	i_ := 0
	i1_ := 0

	//
	// Fast exit (nothing to be done)
	//
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	//
	// Try to call fast SYRK
	//
	if rmatrixsyrkf(n, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper) {
		return
	}

	//
	// SYRK
	//
	if optypea == 0 {
		//
		// C=alpha*A*A^H+beta*C
		//
		for i = 0; i <= n - 1; i++ {
			if isupper {
				j1 = i
				j2 = n - 1
			}else {
				j1 = 0
				j2 = i
			}
			for j = j1; j <= j2; j++ {
				if alpha != 0 && k > 0 {
					v = 0.0
					for i_ = ja; i_ <= ja + k - 1; i_++ {
						v += (*a)[ia + i][ i_] * (*a)[ia + j][ i_]
					}
				}else {
					v = 0
				}
				if beta == 0 {
					(*c)[ic + i][ jc + j] = alpha * v
				}else {
					(*c)[ic + i][ jc + j] = beta * (*c)[ic + i][ jc + j] + alpha * v
				}
			}
		}
		return
	}else {
		//
		// C=alpha*A^H*A+beta*C
		//
		for i = 0; i <= n - 1; i++ {
			if isupper {
				j1 = i
				j2 = n - 1
			}else {
				j1 = 0
				j2 = i
			}
			if beta == 0 {
				for j = j1; j <= j2; j++ {
					(*c)[ic + i][ jc + j] = 0
				}
			}else {
				for i_ = jc + j1; i_ <= jc + j2; i_++ {
					(*c)[ic + i][ i_] = beta * (*c)[ic + i][ i_]
				}
			}
		}
		for i = 0; i <= k - 1; i++ {
			for j = 0; j <= n - 1; j++ {
				if isupper {
					j1 = j
					j2 = n - 1
				}else {
					j1 = 0
					j2 = j
				}
				v = alpha * (*a)[ia + i][ ja + j]
				i1_ = (ja + j1) - (jc + j1)
				for i_ = jc + j1; i_ <= jc + j2; i_++ {
					(*c)[ic + j][ i_] = (*c)[ic + j][ i_] + v * (*a)[ia + i][ i_ + i1_]
				}
			}
		}
		return
	}
}

/*************************************************************************
Same as CMatrixSYRK, but for real matrices

OpType may be only 0 or 1.

  -- ALGLIB routine --
	 16.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixsyrk(n, k int, alpha float64, a *[][]float64, ia, ja, optypea int, beta float64, c *[][]float64, ic, jc int, isupper bool) {
	s1 := 0
	s2 := 0
	bs := 0

	bs = ablasblocksize(a)
	if n <= bs && k <= bs {
		rmatrixsyrk2(n, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
		return
	}
	if k >= n {
		//
		// Split K
		//
		ablassplitlength(a, k, &s1, &s2)
		if optypea == 0 {
			rmatrixsyrk(n, s1, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixsyrk(n, s2, alpha, a, ia, ja + s1, optypea, 1.0, c, ic, jc, isupper)
		}else {
			rmatrixsyrk(n, s1, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixsyrk(n, s2, alpha, a, ia + s1, ja, optypea, 1.0, c, ic, jc, isupper)
		}
	}else {
		//
		// Split N
		//
		ablassplitlength(a, n, &s1, &s2)
		if optypea == 0 && isupper {
			rmatrixsyrk(s1, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixgemm(s1, s2, k, alpha, a, ia, ja, 0, a, ia + s1, ja, 1, beta, c, ic, jc + s1)
			rmatrixsyrk(s2, k, alpha, a, ia + s1, ja, optypea, beta, c, ic + s1, jc + s1, isupper)
			return
		}
		if optypea == 0 && !isupper {
			rmatrixsyrk(s1, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixgemm(s2, s1, k, alpha, a, ia + s1, ja, 0, a, ia, ja, 1, beta, c, ic + s1, jc)
			rmatrixsyrk(s2, k, alpha, a, ia + s1, ja, optypea, beta, c, ic + s1, jc + s1, isupper)
			return
		}
		if optypea != 0 && isupper {
			rmatrixsyrk(s1, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixgemm(s1, s2, k, alpha, a, ia, ja, 1, a, ia, ja + s1, 0, beta, c, ic, jc + s1)
			rmatrixsyrk(s2, k, alpha, a, ia, ja + s1, optypea, beta, c, ic + s1, jc + s1, isupper)
			return
		}
		if optypea != 0 && !isupper {
			rmatrixsyrk(s1, k, alpha, a, ia, ja, optypea, beta, c, ic, jc, isupper)
			rmatrixgemm(s2, s1, k, alpha, a, ia, ja + s1, 1, a, ia, ja, 0, beta, c, ic + s1, jc)
			rmatrixsyrk(s2, k, alpha, a, ia, ja + s1, optypea, beta, c, ic + s1, jc + s1, isupper)
			return
		}
	}
}

/*************************************************************************
Fast kernel

  -- ALGLIB routine --
	 19.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixrighttrsmf(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) bool {
	return false
}

/*************************************************************************
Level 2 subroutine

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixrighttrsm2(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) {
	i := 0
	j := 0
	vr := 0.0
	vd := 0.0
	i_ := 0
	i1_ := 0

	//
	// Special case
	//
	if n * m == 0 {
		return
	}

	//
	// Try to use "fast" code
	//
	if rmatrixrighttrsmf(m, n, a, i1, j1, isupper, isunit, optype, x, i2, j2) {
		return
	}

	//
	// General case
	//
	if isupper {
		//
		// Upper triangular matrix
		//
		if optype == 0 {
			//
			// X*A^(-1)
			//
			for i = 0; i <= m - 1; i++ {
				for j = 0; j <= n - 1; j++ {
					if isunit {
						vd = 1
					}else {
						vd = (*a)[i1 + j][j1 + j]
					}
					(*x)[i2 + i][ j2 + j] = (*x)[i2 + i][ j2 + j] / vd
					if j < n - 1 {
						vr = (*x)[i2 + i][ j2 + j];
						i1_ = (j1 + j + 1) - (j2 + j + 1);
						for i_ = j2 + j + 1; i_ <= j2 + n - 1; i_++ {
							(*x)[i2 + i][ i_] = (*x)[i2 + i][ i_] - vr * (*a)[i1 + j][ i_ + i1_]
						}
					}
				}
			}
			return
		}
		if optype == 1 {
			//
			// X*A^(-T)
			//
			for i = 0; i <= m - 1; i++ {
				for j = n - 1; j >= 0; j-- {
					vr = 0
					vd = 1
					if j < n - 1 {
						i1_ = (j1 + j + 1) - (j2 + j + 1)
						vr = 0.0
						for i_ = j2 + j + 1; i_ <= j2 + n - 1; i_++ {
							vr += (*x)[i2 + i][ i_] * (*a)[i1 + j][ i_ + i1_]
						}
					}
					if !isunit {
						vd = (*a)[i1 + j][ j1 + j]
					}
					(*x)[i2 + i][ j2 + j] = ((*x)[i2 + i][ j2 + j] - vr) / vd
				}
			}
			return
		}
	}else {
		//
		// Lower triangular matrix
		//
		if optype == 0 {
			//
			// X*A^(-1)
			//
			for i = 0; i <= m - 1; i++ {
				for j = n - 1; j >= 0; j-- {
					if isunit {
						vd = 1
					}else {
						vd = (*a)[i1 + j][ j1 + j]
					}
					(*x)[i2 + i][ j2 + j] = (*x)[i2 + i][ j2 + j] / vd
					if j > 0 {
						vr = (*x)[i2 + i][ j2 + j]
						i1_ = (j1) - (j2)
						for i_ = j2; i_ <= j2 + j - 1; i_++ {
							(*x)[i2 + i][ i_] = (*x)[i2 + i][ i_] - vr * (*a)[i1 + j][ i_ + i1_]
						}
					}
				}
			}
			return
		}
		if optype == 1 {
			//
			// X*A^(-T)
			//
			for i = 0; i <= m - 1; i++ {
				for j = 0; j <= n - 1; j++ {
					vr = 0
					vd = 1
					if j > 0 {
						i1_ = (j1) - (j2)
						vr = 0.0
						for i_ = j2; i_ <= j2 + j - 1; i_++ {
							vr += (*x)[i2 + i][ i_] * (*a)[i1 + j][ i_ + i1_]
						}
					}
					if !isunit {
						vd = (*a)[i1 + j][ j1 + j]
					}
					(*x)[i2 + i][ j2 + j] = ((*x)[i2 + i][ j2 + j] - vr) / vd
				}
			}
			return
		}
	}
}

/*************************************************************************
Same as CMatrixRightTRSM, but for real matrices

OpType may be only 0 or 1.

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func rmatrixrighttrsm(m, n int, a *[][]float64, i1, j1 int, isupper, isunit bool, optype int, x *[][]float64, i2, j2 int) {
	s1 := 0
	s2 := 0
	bs := 0

	bs = ablasblocksize(a)
	if m <= bs && n <= bs {
		rmatrixrighttrsm2(m, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
		return
	}
	if m >= n {
		//
		// Split X: X*A = (X1 X2)^T*A
		//
		_i := 2
		ablassplitlength(a, m, &s1, &_i)
		rmatrixrighttrsm(s1, n, a, i1, j1, isupper, isunit, optype, x, i2, j2)
		rmatrixrighttrsm(s2, n, a, i1, j1, isupper, isunit, optype, x, i2 + s1, j2)
	}else {
		//
		// Split A:
		//               (A1  A12)
		// X*op(A) = X*op(       )
		//               (     A2)
		//
		// Different variants depending on
		// IsUpper/OpType combinations
		//
		ablassplitlength(a, n, &s1, &s2)
		if isupper && optype == 0 {
			//
			//                  (A1  A12)-1
			// X*A^-1 = (X1 X2)*(       )
			//                  (     A2)
			//
			rmatrixrighttrsm(m, s1, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			rmatrixgemm(m, s2, s1, -1.0, x, i2, j2, 0, a, i1, j1 + s1, 0, 1.0, x, i2, j2 + s1)
			rmatrixrighttrsm(m, s2, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2, j2 + s1)
			return
		}
		if isupper && optype != 0 {
			//
			//                  (A1'     )-1
			// X*A^-1 = (X1 X2)*(        )
			//                  (A12' A2')
			//
			rmatrixrighttrsm(m, s2, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2, j2 + s1)
			rmatrixgemm(m, s1, s2, -1.0, x, i2, j2 + s1, 0, a, i1, j1 + s1, optype, 1.0, x, i2, j2)
			rmatrixrighttrsm(m, s1, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			return
		}
		if !isupper && optype == 0 {
			//
			//                  (A1     )-1
			// X*A^-1 = (X1 X2)*(       )
			//                  (A21  A2)
			//
			rmatrixrighttrsm(m, s2, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2, j2 + s1)
			rmatrixgemm(m, s1, s2, -1.0, x, i2, j2 + s1, 0, a, i1 + s1, j1, 0, 1.0, x, i2, j2)
			rmatrixrighttrsm(m, s1, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			return
		}
		if !isupper && optype != 0 {
			//
			//                  (A1' A21')-1
			// X*A^-1 = (X1 X2)*(        )
			//                  (     A2')
			//
			rmatrixrighttrsm(m, s1, a, i1, j1, isupper, isunit, optype, x, i2, j2)
			rmatrixgemm(m, s2, s1, -1.0, x, i2, j2, 0, a, i1 + s1, j1, optype, 1.0, x, i2, j2 + s1)
			rmatrixrighttrsm(m, s2, a, i1 + s1, j1 + s1, isupper, isunit, optype, x, i2, j2 + s1)
			return
		}
	}
}

/*************************************************************************
Recursive computational subroutine for SPDMatrixCholesky.

INPUT PARAMETERS:
	A       -   matrix given by upper or lower triangle
	Offs    -   offset of diagonal block to decompose
	N       -   diagonal block size
	IsUpper -   what half is given
	Tmp     -   temporary array; allocated by function, if its size is too
				small; can be reused on subsequent calls.

OUTPUT PARAMETERS:
	A       -   upper (or lower) triangle contains Cholesky decomposition

RESULT:
	True, on success
	False, on failure

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func spdmatrixcholeskyrec(a *[][]float64, offs, n int, isupper bool, tmp *[]float64) bool {
	var result bool
	n1 := 0
	n2 := 0

	//
	// check N
	//
	if n < 1 {
		return false
	}

	//
	// Prepare buffer
	//
	if len(*tmp) < 2 * n {
		*tmp = make([]float64, 2 * n)
	}

	//
	// special cases
	//
	if n == 1 {
		if (*a)[offs][ offs] > 0 {
			(*a)[offs][ offs] = math.Sqrt((*a)[offs][ offs])
			result = true
		}else {
			result = false
		}
		return result
	}
	if n <= ablasblocksize(a) {
		result = spdmatrixcholesky2(a, offs, n, isupper, tmp)
		return result
	}

	//
	// general case: split task in cache-oblivious manner
	//
	result = true
	ablassplitlength(a, n, &n1, &n2)
	result = spdmatrixcholeskyrec(a, offs, n1, isupper, tmp)
	if !result {
		return result
	}
	if n2 > 0 {
		if isupper {
			rmatrixlefttrsm(n1, n2, a, offs, offs, isupper, false, 1, a, offs, offs + n1)
			rmatrixsyrk(n2, n1, -1.0, a, offs, offs + n1, 1, 1.0, a, offs + n1, offs + n1, isupper)
		}else {
			rmatrixrighttrsm(n2, n1, a, offs, offs, isupper, false, 1, a, offs + n1, offs)
			rmatrixsyrk(n2, n1, -1.0, a, offs + n1, offs, 0, 1.0, a, offs + n1, offs + n1, isupper)
		}
		result = spdmatrixcholeskyrec(a, offs + n1, n2, isupper, tmp)
		if !result {
			return result
		}
	}
	return result
}

/*************************************************************************
Cache-oblivious Cholesky decomposition

The algorithm computes Cholesky decomposition  of  a  symmetric  positive-
definite matrix. The result of an algorithm is a representation  of  A  as
A=U^T*U  or A=L*L^T

INPUT PARAMETERS:
	A       -   upper or lower triangle of a factorized matrix.
				array with elements [0..N-1, 0..N-1].
	N       -   size of matrix A.
	IsUpper -   if IsUpper=True, then A contains an upper triangle of
				a symmetric matrix, otherwise A contains a lower one.

OUTPUT PARAMETERS:
	A       -   the result of factorization. If IsUpper=True, then
				the upper triangle contains matrix U, so that A = U^T*U,
				and the elements below the main diagonal are not modified.
				Similarly, if IsUpper = False.

RESULT:
	If  the  matrix  is  positive-definite,  the  function  returns  True.
	Otherwise, the function returns False. Contents of A is not determined
	in such case.

  -- ALGLIB routine --
	 15.12.2009
	 Bochkanov Sergey
*************************************************************************/
func spdmatrixcholesky(a *[][]float64, n int, isupper bool) bool {
	if n < 1 {
		return false
	}
	tmp := make([]float64, 0)
	return spdmatrixcholeskyrec(a, 0, n, isupper, &tmp);
}

/*************************************************************************
Threshold for rcond: matrices with condition number beyond this  threshold
are considered singular.

Threshold must be far enough from underflow, at least Sqr(Threshold)  must
be greater than underflow.
*************************************************************************/
func rcondthreshold() float64 {
	return math.Sqrt(math.Sqrt(minrealnumber))
}

/*************************************************************************
Internal subroutine for matrix norm estimation

  -- LAPACK auxiliary routine (version 3.0) --
	 Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
	 Courant Institute, Argonne National Lab, and Rice University
	 February 29, 1992
*************************************************************************/
func rmatrixestimatenorm(n int, v, x*[]float64, isgn *[]int, est *float64, kase *int) {
	var flg bool
	itmax := 0
	i := 0
	t := 0.0

	positer := 0
	posj := 0
	posjlast := 0
	posjump := 0
	posaltsgn := 0
	posestold := 0
	postemp := 0
	i_ := 0

	itmax = 5
	posaltsgn = n + 1
	posestold = n + 2
	postemp = n + 3
	positer = n + 1
	posj = n + 2
	posjlast = n + 3
	posjump = n + 4
	if *kase == 0 {
		*v = make([]float64, n + 4)
		*x = make([]float64, n + 1)
		*isgn = make([]int, n + 5)
		t = 1 / float64(n)
		for i = 1; i <= n; i++ {
			(*x)[i] = t
		}
		*kase = 1
		(*isgn)[posjump] = 1
		return
	}

	//
	//     ................ ENTRY   (JUMP = 1)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
	//
	if (*isgn)[posjump] == 1 {
		if n == 1 {
			(*v)[1] = (*x)[1]
			*est = math.Abs((*v)[1])
			*kase = 0
			return
		}
		*est = 0
		for i = 1; i <= n; i++ {
			*est = *est + math.Abs((*x)[i])
		}
		for i = 1; i <= n; i++ {
			if (*x)[i] >= 0 {
				(*x)[i] = 1
			}else {
				(*x)[i] = -1
			}
			(*isgn)[i] = utils.SignInt((*x)[i])
		}
		*kase = 2
		(*isgn)[posjump] = 2
		return
	}

	//
	//     ................ ENTRY   (JUMP = 2)
	//     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANDPOSE(A)*X.
	//
	if (*isgn)[posjump] == 2 {
		(*isgn)[posj] = 1
		for i = 2; i <= n; i++ {
			if math.Abs((*x)[i]) > math.Abs((*x)[(*isgn)[posj]]) {
				(*isgn)[posj] = i
			}
		}
		(*isgn)[positer] = 2

		//
		// MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
		//
		for i = 1; i <= n; i++ {
			(*x)[i] = 0
		}
		(*x)[(*isgn)[posj]] = 1
		*kase = 1
		(*isgn)[posjump] = 3
		return
	}

	//
	//     ................ ENTRY   (JUMP = 3)
	//     X HAS BEEN OVERWRITTEN BY A*X.
	//
	if (*isgn)[posjump] == 3 {
		for i_ = 1; i_ <= n; i_++ {
			(*v)[i_] = (*x)[i_]
		}
		(*v)[posestold] = *est
		*est = 0
		for i = 1; i <= n; i++ {
			*est = *est + math.Abs((*v)[i])
		}
		flg = false
		for i = 1; i <= n; i++ {
			if ((*x)[i] >= 0 && (*isgn)[i] < 0) || ((*x)[i] < 0 && (*isgn)[i] >= 0) {
				flg = true
			}
		}

		//
		// REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED.
		// OR MAY BE CYCLING.
		//
		if !flg || *est <= (*v)[posestold] {
			(*v)[posaltsgn] = 1
			for i = 1; i <= n; i++ {
				(*x)[i] = (*v)[posaltsgn] * (1 + float64(i - 1) / float64(n - 1))
				(*v)[posaltsgn] = -(*v)[posaltsgn]
			}
			*kase = 1
			(*isgn)[posjump] = 5
			return
		}
		for i = 1; i <= n; i++ {
			if (*x)[i] >= 0 {
				(*x)[i] = 1
				(*isgn)[i] = 1
			}else {
				(*x)[i] = -1
				(*isgn)[i] = -1
			}
		}
		*kase = 2
		(*isgn)[posjump] = 4
		return
	}

	//
	//     ................ ENTRY   (JUMP = 4)
	//     X HAS BEEN OVERWRITTEN BY TRANDPOSE(A)*X.
	//
	if (*isgn)[posjump] == 4 {
		(*isgn)[posjlast] = (*isgn)[posj]
		(*isgn)[posj] = 1
		for i = 2; i <= n; i++ {
			if math.Abs((*x)[i]) > math.Abs((*x)[(*isgn)[posj]]) {
				(*isgn)[posj] = i
			}
		}
		if (*x)[(*isgn)[posjlast]] != math.Abs((*x)[(*isgn)[posj]]) && (*isgn)[positer] < itmax {
			(*isgn)[positer] = (*isgn)[positer] + 1
			for i = 1; i <= n; i++ {
				(*x)[i] = 0
			}
			(*x)[(*isgn)[posj]] = 1
			*kase = 1
			(*isgn)[posjump] = 3
			return
		}

		//
		// ITERATION COMPLETE.  FINAL STAGE.
		//
		(*v)[posaltsgn] = 1
		for i = 1; i <= n; i++ {
			(*x)[i] = (*v)[posaltsgn] * (1 + float64(i - 1) / float64(n - 1))
			(*v)[posaltsgn] = -(*v)[posaltsgn]
		}
		*kase = 1
		(*isgn)[posjump] = 5
		return
	}

	//
	//     ................ ENTRY   (JUMP = 5)
	//     X HAS BEEN OVERWRITTEN BY A*X.
	//
	if (*isgn)[posjump] == 5 {
		(*v)[postemp] = 0
		for i = 1; i <= n; i++ {
			(*v)[postemp] = (*v)[postemp] + math.Abs((*x)[i])
		}
		(*v)[postemp] = 2 * (*v)[postemp] / float64(3 * n)
		if (*v)[postemp] > *est {
			for i_ = 1; i_ <= n; i_++ {
				(*v)[i_] = (*x)[i_]
			}
			*est = (*v)[postemp]
		}
		*kase = 0
		return
	}
}

/*************************************************************************
complex basic solver-updater for reduced linear system

	alpha*x[i] = beta

solves this equation and updates it in overlfow-safe manner (keeping track
of relative growth of solution).

Parameters:
	Alpha   -   alpha
	Beta    -   beta
	LnMax   -   precomputed Ln(MaxRealNumber)
	BNorm   -   inf-norm of b (right part of original system)
	MaxGrowth-  maximum growth of norm(x) relative to norm(b)
	XNorm   -   inf-norm of other components of X (which are already processed)
				it is updated by CBasicSolveAndUpdate.
	X       -   solution

  -- ALGLIB routine --
	 26.01.2009
	 Bochkanov Sergey
*************************************************************************/
func cbasicsolveandupdate(alpha complex128, beta complex128, lnmax, bnorm, maxgrowth float64, xnorm *float64, x *complex128) bool {
	var result bool
	v := 0.0

	*x = 0

	result = false
	if alpha == 0 {
		return result
	}
	if beta != 0 {
		//
		// alpha*x[i]=beta
		//
		v = math.Log(cmplx.Abs(beta)) - math.Log(cmplx.Abs(alpha))
		if v > lnmax {
			return result
		}
		*x = beta / alpha
	}else {
		//
		// alpha*x[i]=0
		//
		*x = 0
	}

	//
	// update NrmX, test growth limit
	//
	*xnorm = math.Max(*xnorm, cmplx.Abs(*x))
	if *xnorm > maxgrowth * bnorm {
		return result
	}
	result = true
	return result
}

/*************************************************************************
Real implementation of CMatrixScaledTRSafeSolve

  -- ALGLIB routine --
	 21.01.2010
	 Bochkanov Sergey
*************************************************************************/
func rmatrixscaledtrsafesolve(a *[][]float64, sa float64, n int, x *[]float64, isupper bool, trans int, isunit bool, maxgrowth float64) (bool, error) {
	var result bool
	lnmax := 0.0
	nrmb := 0.0
	nrmx := 0.0
	i := 0
	var alpha complex128 = 0
	var beta complex128 = 0
	vr := 0.0
	var cx complex128 = 0
	tmp := make([]float64, 0)
	i_ := 0

	if !(n > 0) {
		return false, fmt.Errorf("RMatrixTRSafeSolve: incorrect N!")
	}
	if !(trans == 0 || trans == 1) {
		return false, fmt.Errorf("RMatrixTRSafeSolve: incorrect Trans!")
	}

	result = true
	lnmax = math.Log(maxrealnumber)

	//
	// Quick return if possible
	//
	if n <= 0 {
		return result, nil
	}

	//
	// Load norms: right part and X
	//
	nrmb = 0
	for i = 0; i <= n - 1; i++ {
		nrmb = math.Max(nrmb, math.Abs((*x)[i]))
	}
	nrmx = 0

	//
	// Solve
	//
	tmp = make([]float64, n)
	result = true
	if isupper && trans == 0 {
		//
		// U*x = b
		//
		for i = n - 1; i >= 0; i-- {
			//
			// Task is reduced to alpha*x[i] = beta
			//
			if isunit {
				alpha = complex(sa, 0)
			}else {
				alpha = complex((*a)[i][ i] * sa, 0)
			}
			if i < n - 1 {
				for i_ = i + 1; i_ <= n - 1; i_++ {
					tmp[i_] = sa * (*a)[i][ i_]
				}
				vr = 0.0
				for i_ = i + 1; i_ <= n - 1; i_++ {
					vr += tmp[i_] * (*x)[i_]
				}
				beta = complex((*x)[i] - vr, 0)
			}else {
				beta = complex((*x)[i], 0)
			}

			//
			// solve alpha*x[i] = beta
			//
			result = cbasicsolveandupdate(alpha, beta, lnmax, nrmb, maxgrowth, &nrmx, &cx)
			if !result {
				return result, nil
			}
			(*x)[i] = real(cx)
		}
		return result, nil
	}
	if !isupper && trans == 0 {
		//
		// L*x = b
		//
		for i = 0; i <= n - 1; i++ {
			//
			// Task is reduced to alpha*x[i] = beta
			//
			if isunit {
				alpha = complex(sa, 0)
			}else {
				alpha = complex((*a)[i][ i] * sa, 0)
			}
			if i > 0 {
				for i_ = 0; i_ <= i - 1; i_++ {
					tmp[i_] = sa * (*a)[i][ i_]
				}
				vr = 0.0
				for i_ = 0; i_ <= i - 1; i_++ {
					vr += tmp[i_] * (*x)[i_]
				}
				beta = complex((*x)[i] - vr, 0)
			}else {
				beta = complex((*x)[i], 0)
			}

			//
			// solve alpha*x[i] = beta
			//
			result = cbasicsolveandupdate(alpha, beta, lnmax, nrmb, maxgrowth, &nrmx, &cx)
			if !result {
				return result, nil
			}
			(*x)[i] = real(cx)
		}
		return result, nil
	}
	if isupper && trans == 1 {
		//
		// U^T*x = b
		//
		for i = 0; i <= n - 1; i++ {
			//
			// Task is reduced to alpha*x[i] = beta
			//
			if isunit {
				alpha = complex(sa, 0)
			}else {
				alpha = complex((*a)[i][ i] * sa, 0)
			}
			beta = complex((*x)[i], 0)

			//
			// solve alpha*x[i] = beta
			//
			result = cbasicsolveandupdate(alpha, beta, lnmax, nrmb, maxgrowth, &nrmx, &cx)
			if !result {
				return result, nil
			}
			(*x)[i] = real(cx)

			//
			// update the rest of right part
			//
			if i < n - 1 {
				vr = real(cx)
				for i_ = i + 1; i_ <= n - 1; i_++ {
					tmp[i_] = sa * (*a)[i][ i_]
				}
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*x)[i_] = (*x)[i_] - vr * tmp[i_]
				}
			}
		}
		return result, nil
	}
	if !isupper && trans == 1 {
		//
		// L^T*x = b
		//
		for i = n - 1; i >= 0; i-- {
			//
			// Task is reduced to alpha*x[i] = beta
			//
			if isunit {
				alpha = complex(sa, 0)
			}else {
				alpha = complex((*a)[i][ i] * sa, 0)
			}
			beta = complex((*x)[i], 0)

			//
			// solve alpha*x[i] = beta
			//
			result = cbasicsolveandupdate(alpha, beta, lnmax, nrmb, maxgrowth, &nrmx, &cx)
			if !result {
				return result, nil
			}
			(*x)[i] = real(cx)

			//
			// update the rest of right part
			//
			if i > 0 {
				vr = real(cx)
				for i_ = 0; i_ <= i - 1; i_++ {
					tmp[i_] = sa * (*a)[i][i_]
				}
				for i_ = 0; i_ <= i - 1; i_++ {
					(*x)[i_] = (*x)[i_] - vr * tmp[i_]
				}
			}
		}
		return result, nil
	}
	result = false
	return result, nil
}

/*************************************************************************
Internal subroutine for condition number estimation

  -- LAPACK routine (version 3.0) --
	 Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
	 Courant Institute, Argonne National Lab, and Rice University
	 February 29, 1992
*************************************************************************/
func spdmatrixrcondcholeskyinternal(cha *[][]float64, n int, isupper, isnormprovided bool, anorm float64, rc *float64) error {
	i := 0
	j := 0
	kase := 0
	ainvnm := 0.0
	ex := make([]float64, 0)
	ev := make([]float64, 0)
	iwork := make([]int, 0)
	sa := 0.0
	v := 0.0
	maxgrowth := 0.0
	i_ := 0
	i1_ := 0


	if !(n >= 1) {
		return fmt.Errorf("ALGLIB: assertion failed")
	}
	tmp := make([]float64, n)

	//
	// RC=0 if something happens
	//
	*rc = 0

	//
	// prepare parameters for triangular solver
	//
	maxgrowth = 1 / rcondthreshold()
	sa = 0
	if isupper {
		for i = 0; i <= n - 1; i++ {
			for j = i; j <= n - 1; j++ {
				sa = math.Max(sa, utils.AbsComplex((*cha)[i][j], 0.0))
			}
		}
	}else {
		for i = 0; i <= n - 1; i++ {
			for j = 0; j <= i; j++ {
				sa = math.Max(sa, utils.AbsComplex((*cha)[i][j], 0.0))
			}
		}
	}
	if sa == 0 {
		sa = 1
	}
	sa = 1 / sa

	//
	// Estimate the norm of A.
	//
	if !isnormprovided {
		kase = 0
		anorm = 0
		for {
			rmatrixestimatenorm(n, &ev, &ex, &iwork, &anorm, &kase)
			if kase == 0 {
				break
			}
			if isupper {
				//
				// Multiply by U
				//
				for i = 1; i <= n; i++ {
					i1_ = (i) - (i - 1)
					v = 0.0
					for i_ = i - 1; i_ <= n - 1; i_++ {
						v += (*cha)[i - 1][ i_] * ex[i_ + i1_]
					}
					ex[i] = v
				}
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = sa * ex[i_]
				}

				//
				// Multiply by U'
				//
				for i = 0; i <= n - 1; i++ {
					tmp[i] = 0
				}
				for i = 0; i <= n - 1; i++ {
					v = ex[i + 1]
					for i_ = i; i_ <= n - 1; i_++ {
						tmp[i_] = tmp[i_] + v * (*cha)[i][ i_]
					}
				}
				i1_ = (0) - (1)
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = tmp[i_ + i1_]
				}
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = sa * ex[i_]
				}
			}else {

				//
				// Multiply by L'
				//
				for i = 0; i <= n - 1; i++ {
					tmp[i] = 0
				}
				for i = 0; i <= n - 1; i++ {
					v = ex[i + 1]
					for i_ = 0; i_ <= i; i_++ {
						tmp[i_] = tmp[i_] + v * (*cha)[i][ i_]
					}
				}
				i1_ = (0) - (1)
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = tmp[i_ + i1_]
				}
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = sa * ex[i_]
				}

				//
				// Multiply by L
				//
				for i = n; i >= 1; i-- {
					i1_ = (1) - (0)
					v = 0.0
					for i_ = 0; i_ <= i - 1; i_++ {
						v += (*cha)[i - 1][ i_] * ex[i_ + i1_]
					}
					ex[i] = v
				}
				for i_ = 1; i_ <= n; i_++ {
					ex[i_] = sa * ex[i_]
				}
			}
		}
	}

	//
	// Quick return if possible
	//
	if anorm == 0 {
		return nil
	}
	if n == 1 {
		*rc = 1
		return nil
	}

	//
	// Estimate the 1-norm of inv(A).
	//
	kase = 0
	for {
		rmatrixestimatenorm(n, &ev, &ex, &iwork, &ainvnm, &kase)
		if kase == 0 {
			break
		}
		for i = 0; i <= n - 1; i++ {
			ex[i] = ex[i + 1]
		}
		if isupper {
			//
			// Multiply by inv(U').
			//
			if res, err := rmatrixscaledtrsafesolve(cha, sa, n, &ex, isupper, 1, false, maxgrowth); !res || err != nil {
				return err
			}

			//
			// Multiply by inv(U).
			//
			if res, err := rmatrixscaledtrsafesolve(cha, sa, n, &ex, isupper, 0, false, maxgrowth); !res || err != nil {
				return err
			}
		}else {

			//
			// Multiply by inv(L).
			//
			if res, err := rmatrixscaledtrsafesolve(cha, sa, n, &ex, isupper, 0, false, maxgrowth); !res || err != nil {
				return err
			}

			//
			// Multiply by inv(L').
			//
			if res, err := rmatrixscaledtrsafesolve(cha, sa, n, &ex, isupper, 1, false, maxgrowth); !res || err != nil {
				return err
			}
		}
		for i = n - 1; i >= 0; i-- {
			ex[i + 1] = ex[i]
		}
	}

	//
	// Compute the estimate of the reciprocal condition number.
	//
	if ainvnm != 0 {
		v = 1 / ainvnm
		*rc = v / anorm
		if *rc < rcondthreshold() {
			*rc = 0
		}
	}
	return nil
}

/*************************************************************************
Condition number estimate of a symmetric positive definite matrix given by
Cholesky decomposition.

The algorithm calculates a lower bound of the condition number. In this
case, the algorithm does not return a lower bound of the condition number,
but an inverse number (to avoid an overflow in case of a singular matrix).

It should be noted that 1-norm and inf-norm condition numbers of symmetric
matrices are equal, so the algorithm doesn't take into account the
differences between these types of norms.

Input parameters:
	CD  - Cholesky decomposition of matrix A,
		  output of SMatrixCholesky subroutine.
	N   - size of matrix A.

Result: 1/LowerBound(cond(A))

NOTE:
	if k(A) is very large, then matrix is  assumed  degenerate,  k(A)=INF,
	0.0 is returned in such cases.
*************************************************************************/
func spdmatrixcholeskyrcond(a *[][]float64, n int, isupper bool) float64 {
	result := 0.0
	spdmatrixrcondcholeskyinternal(a, n, isupper, false, 0, &result);
	return result;
}

/*************************************************************************
Basic Cholesky solver for ScaleA*Cholesky(A)'*x = y.

This subroutine assumes that:
* A*ScaleA is well scaled
* A is well-conditioned, so no zero divisions or overflow may occur

  -- ALGLIB --
	 Copyright 27.01.2010 by Bochkanov Sergey
*************************************************************************/
func spdbasiccholeskysolve(cha *[][]float64, sqrtscalea float64, n int, isupper bool, xb, tmp *[]float64) {
	i := 0
	v := 0.0
	i_ := 0

	//
	// A = L*L' or A=U'*U
	//
	if isupper {
		//
		// Solve U'*y=b first.
		//
		for i = 0; i <= n - 1; i++ {
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][i])
			if i < n - 1 {
				v = (*xb)[i]
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][i_]
				}
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*xb)[i_] = (*xb)[i_] - v * (*tmp)[i_]
				}
			}
		}

		//
		// Solve U*x=y then.
		//
		for i = n - 1; i >= 0; i-- {
			if i < n - 1 {
				for i_ = i + 1; i_ <= n - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				v = 0.0
				for i_ = i + 1; i_ <= n - 1; i_++ {
					v += (*tmp)[i_] * (*xb)[i_]
				}
				(*xb)[i] = (*xb)[i] - v
			}
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
		}
	}else {
		//
		// Solve L*y=b first
		//
		for i = 0; i <= n - 1; i++ {
			if i > 0 {
				for i_ = 0; i_ <= i - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				v = 0.0
				for i_ = 0; i_ <= i - 1; i_++ {
					v += (*tmp)[i_] * (*xb)[i_]
				}
				(*xb)[i] = (*xb)[i] - v
			}
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
		}

		//
		// Solve L'*x=y then.
		//
		for i = n - 1; i >= 0; i-- {
			(*xb)[i] = (*xb)[i] / (sqrtscalea * (*cha)[i][ i])
			if i > 0 {
				v = (*xb)[i]
				for i_ = 0; i_ <= i - 1; i_++ {
					(*tmp)[i_] = sqrtscalea * (*cha)[i][ i_]
				}
				for i_ = 0; i_ <= i - 1; i_++ {
					(*xb)[i_] = (*xb)[i_] - v * (*tmp)[i_]
				}
			}
		}
	}
}

/*************************************************************************
Internal Cholesky solver

  -- ALGLIB --
	 Copyright 27.01.2010 by Bochkanov Sergey
*************************************************************************/
func spdmatrixcholeskysolveinternal(cha *[][]float64, sqrtscalea float64, n int, isupper bool, a *[][]float64, havea bool, b *[][]float64, m int, info *int, rep *densesolverreport, x *[][]float64) error {
	i := 0
	j := 0
	k := 0
	//	y := [0]float64
	//	xa := [0]float64
	//	xb := [0]float64
	v := 0.0
	mxb := 0.0
	scaleright := 0.0
	i_ := 0

	*info = 0

	if !(sqrtscalea > 0) {
		return fmt.Errorf("ALGLIB: assertion failed")
	}

	//
	// prepare: check inputs, allocate space...
	//
	if n <= 0 || m <= 0 {
		*info = -1
		return nil
	}
	*x = utils.MakeMatrixFloat64(n, m)
	//	y = [n]float64
	xc := make([]float64, n)
	bc := make([]float64, n)
	tx := make([]float64, n + 1)
	//	xa = [n + 1]float64
	//	xb = [n + 1]float64

	//
	// estimate condition number, test for near singularity
	//
	rep.r1 = spdmatrixcholeskyrcond(cha, n, isupper)
	rep.rinf = rep.r1
	if rep.r1 < rcondthreshold() {
		for i = 0; i <= n - 1; i++ {
			for j = 0; j <= m - 1; j++ {
				(*x)[i][ j] = 0
			}
		}
		rep.r1 = 0
		rep.rinf = 0
		*info = -3
		return nil
	}
	*info = 1

	//
	// solve
	//
	for k = 0; k <= m - 1; k++ {
		//
		// copy B to contiguous storage
		//
		for i_ = 0; i_ <= n - 1; i_++ {
			bc[i_] = (*b)[i_][ k]
		}

		//
		// Scale right part:
		// * MX stores max(|Bi|)
		// * ScaleRight stores actual scaling applied to B when solving systems
		//   it is chosen to make |scaleRight*b| close to 1.
		//
		mxb = 0
		for i = 0; i <= n - 1; i++ {
			mxb = math.Max(mxb, math.Abs(bc[i]))
		}
		if mxb == 0 {
			mxb = 1
		}
		scaleright = 1 / mxb

		//
		// First, non-iterative part of solution process.
		// We use separate code for this task because
		// XDot is quite slow and we want to save time.
		//
		for i_ = 0; i_ <= n - 1; i_++ {
			xc[i_] = scaleright * bc[i_]
		}
		spdbasiccholeskysolve(cha, sqrtscalea, n, isupper, &xc, &tx)

		//
		// Store xc.
		// Post-scale result.
		//
		v = (sqrtscalea * sqrtscalea) * mxb
		for i_ = 0; i_ <= n - 1; i_++ {
			(*x)[i_][ k] = v * xc[i_]
		}
	}
	return nil
}

/*************************************************************************
Dense solver. Same as RMatrixLUSolveM(), but for SPD matrices  represented
by their Cholesky decomposition.

Algorithm features:
* automatic detection of degenerate cases
* O(M*N^2) complexity
* condition number estimation
* matrix is represented by its upper or lower triangle

No iterative refinement is provided because such partial representation of
matrix does not allow efficient calculation of extra-precise  matrix-vector
products for large matrices. Use RMatrixSolve or RMatrixMixedSolve  if  you
need iterative refinement.

INPUT PARAMETERS
	CHA     -   array[0..N-1,0..N-1], Cholesky decomposition,
				SPDMatrixCholesky result
	N       -   size of CHA
	IsUpper -   what half of CHA is provided
	B       -   array[0..N-1,0..M-1], right part
	M       -   right part size

OUTPUT PARAMETERS
	Info    -   same as in RMatrixSolve
	Rep     -   same as in RMatrixSolve
	X       -   same as in RMatrixSolve

  -- ALGLIB --
	 Copyright 27.01.2010 by Bochkanov Sergey
*************************************************************************/
func spdmatrixcholeskysolvem(cha *[][]float64, n int, isupper bool, b *[][]float64, m int, info *int, rep *densesolverreport, x *[][]float64) {
	emptya := utils.MakeMatrixFloat64(0, 0)
	sqrtscalea := 0.0
	i := 0
	j := 0
	j1 := 0
	j2 := 0

	*info = 0
	*x = utils.MakeMatrixFloat64(0, 0)

	//
	// prepare: check inputs, allocate space...
	//
	if n <= 0 || m <= 0 {
		*info = -1
		return
	}

	//
	// 1. scale matrix, max(|U[i,j]|)
	// 2. factorize scaled matrix
	// 3. solve
	//
	sqrtscalea = 0
	for i = 0; i <= n - 1; i++ {
		if isupper {
			j1 = i
			j2 = n - 1
		}else {
			j1 = 0
			j2 = i
		}
		for j = j1; j <= j2; j++ {
			sqrtscalea = math.Max(sqrtscalea, math.Abs((*cha)[i][ j]))
		}
	}
	if sqrtscalea == 0 {
		sqrtscalea = 1
	}
	sqrtscalea = 1 / sqrtscalea
	spdmatrixcholeskysolveinternal(cha, sqrtscalea, n, isupper, &emptya, false, b, m, info, rep, x)
}

/*************************************************************************
Dense solver. Same as RMatrixLUSolve(), but for  SPD matrices  represented
by their Cholesky decomposition.

Algorithm features:
* automatic detection of degenerate cases
* O(N^2) complexity
* condition number estimation
* matrix is represented by its upper or lower triangle

No iterative refinement is provided because such partial representation of
matrix does not allow efficient calculation of extra-precise  matrix-vector
products for large matrices. Use RMatrixSolve or RMatrixMixedSolve  if  you
need iterative refinement.

INPUT PARAMETERS
	CHA     -   array[0..N-1,0..N-1], Cholesky decomposition,
				SPDMatrixCholesky result
	N       -   size of A
	IsUpper -   what half of CHA is provided
	B       -   array[0..N-1], right part

OUTPUT PARAMETERS
	Info    -   same as in RMatrixSolve
	Rep     -   same as in RMatrixSolve
	X       -   same as in RMatrixSolve

  -- ALGLIB --
	 Copyright 27.01.2010 by Bochkanov Sergey
*************************************************************************/
func spdmatrixcholeskysolve(cha *[][]float64, n int, isupper bool, b *[]float64, info *int, rep *densesolverreport, x *[]float64) {
	xm := utils.MakeMatrixFloat64(0, 0)
	i_ := 0
	*info = 0
	if n <= 0 {
		*info = -1;
		return;
	}
	bm := utils.MakeMatrixFloat64(n, 1)
	for i_ = 0; i_ <= n - 1; i_++ {
		bm[i_][ 0] = (*b)[i_]
	}
	spdmatrixcholeskysolvem(cha, n, isupper, &bm, 1, info, rep, &xm)
	*x = make([]float64, n)
	for i_ = 0; i_ <= n - 1; i_++ {
		(*x)[i_] = xm[i_][0]
	}
}

/*************************************************************************
This function checks that all values from upper/lower triangle of
X[0..N-1,0..N-1] are finite

  -- ALGLIB --
	 Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
func isfinitertrmatrix(x *[][]float64, n int, isupper bool) (bool, error) {
	i := 0
	j1 := 0
	j2 := 0
	j := 0

	if !(n >= 0) {
		return false, fmt.Errorf("APSERVIsFiniteRTRMatrix: internal error (N<0)")
	}
	for i = 0; i <= n - 1; i++ {
		if isupper {
			j1 = i
			j2 = n - 1
		}else {
			j1 = 0
			j2 = i
		}
		for j = j1; j <= j2; j++ {
			if !utils.IsFinite((*x)[i][j]) {
				return false, nil
			}
		}
	}
	return true, nil
}

/*************************************************************************
Internal subroutine for condition number estimation

  -- LAPACK routine (version 3.0) --
	 Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
	 Courant Institute, Argonne National Lab, and Rice University
	 February 29, 1992
*************************************************************************/
func rmatrixrcondtrinternal(a *[][]float64, n int, isupper, isunit, onenorm bool, anorm float64, rc *float64) {
	ex := make([]float64, 0)
	ev := make([]float64, 0)
	i := 0
	j := 0
	kase := 0
	kase1 := 0
	j1 := 0
	j2 := 0
	ainvnm := 0.0
	maxgrowth := 0.0
	s := 0.0

	//
	// RC=0 if something happens
	//
	*rc = 0

	//
	// init
	//
	if onenorm {
		kase1 = 1
	}else {
		kase1 = 2
	}
	//	mupper := true
	//	mtra0ns := true
	//	munit := true
	iwork := make([]int, n + 1)
	//	tmp := [n]float64

	//
	// prepare parameters for triangular solver
	//
	maxgrowth = 1 / rcondthreshold()
	s = 0
	for i = 0; i <= n - 1; i++ {
		if isupper {
			j1 = i + 1
			j2 = n - 1
		}else {
			j1 = 0
			j2 = i - 1
		}
		for j = j1; j <= j2; j++ {
			s = math.Max(s, math.Abs((*a)[i][ j]))
		}
		if isunit {
			s = math.Max(s, 1)
		}else {
			s = math.Max(s, math.Abs((*a)[i][ i]))
		}
	}
	if s == 0 {
		s = 1
	}
	s = 1 / s

	//
	// Scale according to S
	//
	anorm = anorm * s

	//
	// Quick return if possible
	// We assume that ANORM<>0 after this block
	//
	if anorm == 0 {
		return
	}
	if n == 1 {
		*rc = 1
		return
	}

	//
	// Estimate the norm of inv(A).
	//
	ainvnm = 0
	kase = 0
	for {
		rmatrixestimatenorm(n, &ev, &ex, &iwork, &ainvnm, &kase)
		if kase == 0 {
			break
		}

		//
		// from 1-based array to 0-based
		//
		for i = 0; i <= n - 1; i++ {
			ex[i] = ex[i + 1]
		}

		//
		// multiply by inv(A) or inv(A')
		//
		if kase == kase1 {
			//
			// multiply by inv(A)
			//
			if res, _ := rmatrixscaledtrsafesolve(a, s, n, &ex, isupper, 0, isunit, maxgrowth); !res {
				return
			}
		}else {
			//
			// multiply by inv(A')
			//
			if res, _ := rmatrixscaledtrsafesolve(a, s, n, &ex, isupper, 1, isunit, maxgrowth); !res {
				return
			}
		}

		//
		// from 0-based array to 1-based
		//
		for i = n - 1; i >= 0; i-- {
			ex[i + 1] = ex[i]
		}
	}

	//
	// Compute the estimate of the reciprocal condition number.
	//
	if ainvnm != 0 {
		*rc = 1 / ainvnm
		*rc = *rc / anorm
		if *rc < rcondthreshold() {
			*rc = 0
		}
	}
}

/*************************************************************************
Triangular matrix: estimate of a condition number (1-norm)

The algorithm calculates a lower bound of the condition number. In this case,
the algorithm does not return a lower bound of the condition number, but an
inverse number (to avoid an overflow in case of a singular matrix).

Input parameters:
	A       -   matrix. Array[0..N-1, 0..N-1].
	N       -   size of A.
	IsUpper -   True, if the matrix is upper triangular.
	IsUnit  -   True, if the matrix has a unit diagonal.

Result: 1/LowerBound(cond(A))

NOTE:
	if k(A) is very large, then matrix is  assumed  degenerate,  k(A)=INF,
	0.0 is returned in such cases.
*************************************************************************/
func rmatrixtrrcond1(a *[][]float64, n int, isupper, isunit bool) (float64, error) {
	result := 0.0
	i := 0
	j := 0
	nrm := 0.0
	//	pivots := [0]int
	j1 := 0
	j2 := 0

	if !(n >= 1) {
		return 0, fmt.Errorf("RMatrixTRRCond1: N<1!")
	}
	t := make([]float64, n)
	for i = 0; i <= n - 1; i++ {
		t[i] = 0
	}
	for i = 0; i <= n - 1; i++ {
		if isupper {
			j1 = i + 1
			j2 = n - 1
		}else {
			j1 = 0
			j2 = i - 1
		}
		for j = j1; j <= j2; j++ {
			t[j] = t[j] + math.Abs((*a)[i][ j])
		}
		if isunit {
			t[i] = t[i] + 1
		}else {
			t[i] = t[i] + math.Abs((*a)[i][ i])
		}
	}
	nrm = 0
	for i = 0; i <= n - 1; i++ {
		nrm = math.Max(nrm, t[i])
	}
	rmatrixrcondtrinternal(a, n, isupper, isunit, true, nrm, &result)
	return result, nil
}

/*************************************************************************
Triangular matrix: estimate of a matrix condition number (infinity-norm).

The algorithm calculates a lower bound of the condition number. In this case,
the algorithm does not return a lower bound of the condition number, but an
inverse number (to avoid an overflow in case of a singular matrix).

Input parameters:
	A   -   matrix. Array whose indexes range within [0..N-1, 0..N-1].
	N   -   size of matrix A.
	IsUpper -   True, if the matrix is upper triangular.
	IsUnit  -   True, if the matrix has a unit diagonal.

Result: 1/LowerBound(cond(A))

NOTE:
	if k(A) is very large, then matrix is  assumed  degenerate,  k(A)=INF,
	0.0 is returned in such cases.
*************************************************************************/
func rmatrixtrrcondinf(a *[][]float64, n int, isupper, isunit bool) (float64, error) {
	result := 0.0
	i := 0
	j := 0
	v := 0.0
	nrm := 0.0
	//	pivots := [0]int
	j1 := 0
	j2 := 0

	if !(n >= 1) {
		return 0, fmt.Errorf("RMatrixTRRCondInf: N<1!")
	}

	nrm = 0
	for i = 0; i <= n - 1; i++ {
		if isupper {
			j1 = i + 1
			j2 = n - 1
		}else {
			j1 = 0
			j2 = i - 1
		}
		v = 0
		for j = j1; j <= j2; j++ {
			v = v + math.Abs((*a)[i][ j])
		}
		if isunit {
			v = v + 1
		}else {
			v = v + math.Abs((*a)[i][ i])
		}
		nrm = math.Max(nrm, v)
	}
	rmatrixrcondtrinternal(a, n, isupper, isunit, false, nrm, &v)
	result = v
	return result, nil
}

/*************************************************************************
Triangular matrix inversion, recursive subroutine

  -- ALGLIB --
	 05.02.2010, Bochkanov Sergey.
	 Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
	 Courant Institute, Argonne National Lab, and Rice University
	 February 29, 1992.
*************************************************************************/
func rmatrixtrinverserec(a *[][]float64, offs, n int, isupper, isunit bool, tmp *[]float64, info *int, rep *matinvreport) {
	n1 := 0
	n2 := 0
	i := 0
	j := 0
	v := 0.0
	ajj := 0.0
	i_ := 0
	i1_ := 0

	if n < 1 {
		*info = -1
		return
	}

	//
	// Base case
	//
	if n <= ablasblocksize(a) {
		if isupper {
			//
			// Compute inverse of upper triangular matrix.
			//
			for j = 0; j <= n - 1; j++ {
				if !isunit {
					if (*a)[offs + j][ offs + j] == 0 {
						*info = -3
						return
					}
					(*a)[offs + j][ offs + j] = 1 / (*a)[offs + j][offs + j]
					ajj = -(*a)[offs + j][ offs + j]
				}else {
					ajj = -1
				}

				//
				// Compute elements 1:j-1 of j-th column.
				//
				if j > 0 {
					i1_ = (offs + 0) - (0)
					for i_ = 0; i_ <= j - 1; i_++ {
						(*tmp)[i_] = (*a)[i_ + i1_][ offs + j]
					}
					for i = 0; i <= j - 1; i++ {
						if i < j - 1 {
							i1_ = (i + 1) - (offs + i + 1)
							v = 0.0
							for i_ = offs + i + 1; i_ <= offs + j - 1; i_++ {
								v += (*a)[offs + i][ i_] * (*tmp)[i_ + i1_]
							}
						}else {
							v = 0
						}
						if !isunit {
							(*a)[offs + i][ offs + j] = v + (*a)[offs + i][ offs + i] * (*tmp)[i]
						}else {
							(*a)[offs + i][ offs + j] = v + (*tmp)[i]
						}
					}
					for i_ = offs + 0; i_ <= offs + j - 1; i_++ {
						(*a)[i][ offs + j] = ajj * (*a)[i_][ offs + j]
					}
				}
			}
		}else {
			//
			// Compute inverse of lower triangular matrix.
			//
			for j = n - 1; j >= 0; j-- {
				if !isunit {
					if (*a)[offs + j][ offs + j] == 0 {
						*info = -3
						return
					}
					(*a)[offs + j][ offs + j] = 1 / (*a)[offs + j][offs + j]
					ajj = -(*a)[offs + j][ offs + j]
				}else {
					ajj = -1
				}
				if j < n - 1 {
					//
					// Compute elements j+1:n of j-th column.
					//
					i1_ = (offs + j + 1) - (j + 1)
					for i_ = j + 1; i_ <= n - 1; i_++ {
						(*tmp)[i_] = (*a)[i_ + i1_][ offs + j]
					}
					for i = j + 1; i <= n - 1; i++ {
						if i > j + 1 {
							i1_ = (j + 1) - (offs + j + 1)
							v = 0.0
							for i_ = offs + j + 1; i_ <= offs + i - 1; i_++ {
								v += (*a)[offs + i][ i_] * (*tmp)[i_ + i1_]
							}
						}else {
							v = 0
						}
						if !isunit {
							(*a)[offs + i][ offs + j] = v + (*a)[offs + i][ offs + i] * (*tmp)[i]
						}else {
							(*a)[offs + i][ offs + j] = v + (*tmp)[i]
						}
					}
					for i_ = offs + j + 1; i_ <= offs + n - 1; i_++ {
						(*a)[i_][ offs + j] = ajj * (*a)[i_][ offs + j]
					}
				}
			}
		}
		return
	}

	//
	// Recursive case
	//
	ablassplitlength(a, n, &n1, &n2)
	if n2 > 0 {
		if isupper {
			for i = 0; i <= n1 - 1; i++ {
				for i_ = offs + n1; i_ <= offs + n - 1; i_++ {
					(*a)[offs + i][ i_] = -1 * (*a)[offs + i][ i_]
				}
			}
			rmatrixlefttrsm(n1, n2, a, offs, offs, isupper, isunit, 0, a, offs, offs + n1)
			rmatrixrighttrsm(n1, n2, a, offs + n1, offs + n1, isupper, isunit, 0, a, offs, offs + n1)
		}else {
			for i = 0; i <= n2 - 1; i++ {
				for i_ = offs; i_ <= offs + n1 - 1; i_++ {
					(*a)[offs + n1 + i][ i_] = -1 * (*a)[offs + n1 + i][ i_]
				}
			}
			rmatrixrighttrsm(n2, n1, a, offs, offs, isupper, isunit, 0, a, offs + n1, offs)
			rmatrixlefttrsm(n2, n1, a, offs + n1, offs + n1, isupper, isunit, 0, a, offs + n1, offs)
		}
		rmatrixtrinverserec(a, offs + n1, n2, isupper, isunit, tmp, info, rep)
	}
	rmatrixtrinverserec(a, offs, n1, isupper, isunit, tmp, info, rep)
}

/*************************************************************************
Triangular matrix inverse (real)

The subroutine inverts the following types of matrices:
	* upper triangular
	* upper triangular with unit diagonal
	* lower triangular
	* lower triangular with unit diagonal

In case of an upper (lower) triangular matrix,  the  inverse  matrix  will
also be upper (lower) triangular, and after the end of the algorithm,  the
inverse matrix replaces the source matrix. The elements  below (above) the
main diagonal are not changed by the algorithm.

If  the matrix  has a unit diagonal, the inverse matrix also  has  a  unit
diagonal, and the diagonal elements are not passed to the algorithm.

Input parameters:
	A       -   matrix, array[0..N-1, 0..N-1].
	N       -   size of matrix A (optional) :
				* if given, only principal NxN submatrix is processed  and
				  overwritten. other elements are unchanged.
				* if not given,  size  is  automatically  determined  from
				  matrix size (A must be square matrix)
	IsUpper -   True, if the matrix is upper triangular.
	IsUnit  -   diagonal type (optional):
				* if True, matrix has unit diagonal (a[i,i] are NOT used)
				* if False, matrix diagonal is arbitrary
				* if not given, False is assumed

Output parameters:
	Info    -   same as for RMatrixLUInverse
	Rep     -   same as for RMatrixLUInverse
	A       -   same as for RMatrixLUInverse.

  -- ALGLIB --
	 Copyright 05.02.2010 by Bochkanov Sergey
*************************************************************************/
func rmatrixtrinverse(a *[][]float64, n int, isupper, isunit bool, info *int, rep *matinvreport) error {
	i := 0
	j := 0

	*info = 0

	if !(n > 0) {
		return fmt.Errorf("RMatrixTRInverse: N<=0!")
	}
	if !(len(*a) >= n ) {
		return fmt.Errorf("RMatrixTRInverse: cols(A)<N!")
	}
	if !(len((*a)[0]) >= n) {
		return fmt.Errorf("RMatrixTRInverse: rows(A)<N!")
	}
	if res, _ := isfinitertrmatrix(a, n, isupper); !res {
		return fmt.Errorf("RMatrixTRInverse: A contains infinite or NaN values!")
	}
	*info = 1

	//
	// calculate condition numbers
	//
	var err error
	if rep.r1, err = rmatrixtrrcond1(a, n, isupper, isunit); err != nil {
		return err
	}
	if rep.rinf, err = rmatrixtrrcondinf(a, n, isupper, isunit); err != nil {
		return err
	}
	if rep.r1 < rcondthreshold() || rep.rinf < rcondthreshold() {
		for i = 0; i <= n - 1; i++ {
			for j = 0; j <= n - 1; j++ {
				(*a)[i][ j] = 0
			}
		}
		rep.r1 = 0
		rep.rinf = 0
		*info = -3
		return nil
	}

	//
	// Invert
	//
	tmp := make([]float64, n)
	rmatrixtrinverserec(a, 0, n, isupper, isunit, &tmp, info, rep)
	return nil
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
	g := make([]float64, 0)
	//	d := [0]float64
	h := utils.MakeMatrixFloat64(0, 0)
	hmod := utils.MakeMatrixFloat64(0, 0)
	//	z := [0][0]float64
	nu := 0.0
	lambdav := 0.0
	internalrep := &minlbfgsreport{}
	state := NewMinLbfgsState()
	//	x := [0]float64
	//	y := [0]float64
	wbase := make([]float64, 0)
	wdir := make([]float64, 0)
	wt := make([]float64, 0)
	//	wx := [0]float64
	pass := 0
	wbest := make([]float64, 0)
	ebest := 0.0
	invinfo := 0
	invrep := &matinvreport{}
	solverinfo := 0
	solverrep := &densesolverreport{}
	i_ := 0

	*info = 0

	mlpbase.MlpProperties(network, &nin, &nout, &wcount)
	lambdaup := 10.0
	lambdadown := 0.3
	//	lmftol := 0.001
	lmsteptol := 0.001

	//
	// Test for inputs
	//
	if npoints <= 0 || restarts < 1 {
		*info = -1
		return
	}
	if mlpbase.MlpIsSoftMax(network) {
		for i = 0; i <= npoints - 1; i++ {
			if utils.RoundInt((*xy)[i][ nin]) < 0 || utils.RoundInt((*xy)[i][nin]) >= nout {
				*info = -2
				return
			}
		}
	}
	decay = math.Max(decay, mindecay)
	*info = 2

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
	mlpbase.MlpInitPreprocessor(network, *xy, npoints)
	g = make([]float64, wcount - 1 + 1)
	h = utils.MakeMatrixFloat64(wcount - 1 + 1, wcount - 1 + 1)
	hmod = utils.MakeMatrixFloat64(wcount - 1 + 1, wcount - 1 + 1)
	wbase = make([]float64, wcount - 1 + 1)
	wdir = make([]float64, wcount - 1 + 1)
	wbest = make([]float64, wcount - 1 + 1)
	wt = make([]float64, wcount - 1 + 1)
	//	wx = [wcount - 1 + 1]float64
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
		minlbfgscreate(wcount, utils.MinInt(wcount, 5), &wbase, state)
		minlbfgssetcond(state, 0, 0, 0, utils.MaxInt(25, wcount))
		for minlbfgsiteration(state) {
			//
			// gradient
			//
			for i_ = 0; i_ <= wcount - 1; i_++ {
				network.Weights[i_] = state.x[i_]
			}
			mlpbase.MlpGradBatch(network, *xy, npoints, &state.f, &state.g)

			//
			// weight decay
			//
			v = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				v += network.Weights[i_] * network.Weights[i_]
			}
			state.f = state.f + 0.5 * decay * v
			for i_ = 0; i_ <= wcount - 1; i_++ {
				state.g[i_] = state.g[i_] + decay * network.Weights[i_]
			}

			//
			// next iteration
			//
			rep.NGrad = rep.NGrad + 1
		}
		minlbfgsresults(state, &wbase, internalrep)
		for i_ = 0; i_ <= wcount - 1; i_++ {
			network.Weights[i_] = wbase[i_]
		}

		//
		// Second stage of the hybrid algorithm: LM
		//
		// Initialize H with identity matrix,
		// G with gradient,
		// E with regularized error.
		//
		mlpbase.MlpHessianBatch(network, xy, npoints, &e, &g, &h)
		v = 0.0
		for i_ = 0; i_ <= wcount - 1; i_++ {
			v += network.Weights[i_] * network.Weights[i_]
		}
		e = e + 0.5 * decay * v
		for i_ = 0; i_ <= wcount - 1; i_++ {
			g[i_] = g[i_] + decay * network.Weights[i_]
		}
		for k = 0; k <= wcount - 1; k++ {
			h[k][ k] = h[k][ k] + decay
		}
		rep.NHess = rep.NHess + 1
		lambdav = 0.001
		nu = 2
		for {
			//
			// 1. HMod = H+lambda*I
			// 2. Try to solve (H+Lambda*I)*dx = -g.
			//    Increase lambda if left part is not positive definite.
			//
			for i = 0; i <= wcount - 1; i++ {
				for i_ = 0; i_ <= wcount - 1; i_++ {
					hmod[i][ i_] = h[i][ i_]
				}
				hmod[i][ i] = hmod[i][ i] + lambdav
			}
			spd := spdmatrixcholesky(&hmod, wcount, true)
			rep.NCholesky = rep.NCholesky + 1
			if !spd {
				lambdav = lambdav * lambdaup * nu
				nu = nu * 2
				continue
			}
			spdmatrixcholeskysolve(&hmod, wcount, true, &g, &solverinfo, solverrep, &wdir)
			if solverinfo < 0 {
				lambdav = lambdav * lambdaup * nu
				nu = nu * 2
				continue
			}
			for i_ = 0; i_ <= wcount - 1; i_++ {
				wdir[i_] = -1 * wdir[i_]
			}

			//
			// Lambda found.
			// 1. Save old w in WBase
			// 1. Test some stopping criterions
			// 2. If error(w+wdir)>error(w), increase lambda
			//
			for i_ = 0; i_ <= wcount - 1; i_++ {
				network.Weights[i_] = network.Weights[i_] + wdir[i_]
			}
			xnorm2 = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				xnorm2 += network.Weights[i_] * network.Weights[i_]
			}
			stepnorm = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				stepnorm += wdir[i_] * wdir[i_]
			}
			stepnorm = math.Sqrt(stepnorm);
			enew = mlpbase.MlpError(network, xy, npoints) + 0.5 * decay * xnorm2
			if stepnorm < lmsteptol * (1 + math.Sqrt(xnorm2)) {
				break
			}
			if enew > e {
				lambdav = lambdav * lambdaup * nu
				nu = nu * 2
				continue
			}

			//
			// Optimize using inv(cholesky(H)) as preconditioner
			//
			rmatrixtrinverse(&hmod, wcount, true, false, &invinfo, invrep)
			if invinfo <= 0 {
				//
				// if matrix can't be inverted then exit with errors
				// TODO: make WCount steps in direction suggested by HMod
				//
				*info = -9
				return
			}
			for i_ = 0; i_ <= wcount - 1; i_++ {
				wbase[i_] = network.Weights[i_]
			}
			for i = 0; i <= wcount - 1; i++ {
				wt[i] = 0
			}
			if err := minlbfgscreatex(wcount, wcount, &wt, 1, 0.0, state); err != nil {
				fmt.Printf("Err 01: %v", err)
			}
			minlbfgssetcond(state, 0, 0, 0, 5)
			for minlbfgsiteration(state) {
				//
				// gradient
				//
				for i = 0; i <= wcount - 1; i++ {
					v = 0.0
					for i_ = i; i_ <= wcount - 1; i_++ {
						v += state.x[i_] * hmod[i][ i_]
					}
					network.Weights[i] = wbase[i] + v
				}
				mlpbase.MlpGradBatch(network, *xy, npoints, &state.f, &g)
				for i = 0; i <= wcount - 1; i++ {
					state.g[i] = 0
				}
				for i = 0; i <= wcount - 1; i++ {
					v = g[i]
					for i_ = i; i_ <= wcount - 1; i_++ {
						state.g[i_] = state.g[i_] + v * hmod[i][i_]
					}
				}

				//
				// weight decay
				// grad(x'*x) = A'*(x0+A*t)
				//
				v = 0.0
				for i_ = 0; i_ <= wcount - 1; i_++ {
					v += network.Weights[i_] * network.Weights[i_]
				}
				state.f = state.f + 0.5 * decay * v
				for i = 0; i <= wcount - 1; i++ {
					v = decay * network.Weights[i]
					for i_ = i; i_ <= wcount - 1; i_++ {
						state.g[i_] = state.g[i_] + v * hmod[i][i_]
					}
				}

				//
				// next iteration
				//
				rep.NGrad += 1
			}
			minlbfgsresults(state, &wt, internalrep)

			//
			// Accept new position.
			// Calculate Hessian
			//
			for i = 0; i <= wcount - 1; i++ {
				v = 0.0
				for i_ = i; i_ <= wcount - 1; i_++ {
					v += wt[i_] * hmod[i][ i_]
				}
				network.Weights[i] = wbase[i] + v
			}
			mlpbase.MlpHessianBatch(network, xy, npoints, &e, &g, &h)
			v = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				v += network.Weights[i_] * network.Weights[i_]
			}
			e = e + 0.5 * decay * v
			for i_ = 0; i_ <= wcount - 1; i_++ {
				g[i_] = g[i_] + decay * network.Weights[i_]
			}
			for k = 0; k <= wcount - 1; k++ {
				h[k][ k] = h[k][ k] + decay
			}
			rep.NHess += 1

			//
			// Update lambda
			//
			lambdav = lambdav * lambdadown
			nu = 2
		}

		//
		// update WBest
		//
		v = 0.0
		for i_ = 0; i_ <= wcount - 1; i_++ {
			v += network.Weights[i_] * network.Weights[i_]
		}
		e = 0.5 * decay * v + mlpbase.MlpError(network, xy, npoints)
		if e < ebest {
			ebest = e
			for i_ = 0; i_ <= wcount - 1; i_++ {
				wbest[i_] = network.Weights[i_]
			}
		}
	}

	//
	// copy WBest to output
	//
	for i_ = 0; i_ <= wcount - 1; i_++ {
		network.Weights[i_] = wbest[i_]
	}
}

/*************************************************************************
Neural  network  training  using  L-BFGS  algorithm  with  regularization.
Subroutine  trains  neural  network  with  restarts from random positions.
Algorithm  is  well  suited  for  problems  of  any dimensionality (memory
requirements and step complexity are linear by weights number).

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
	WStep       -   stopping criterion. Algorithm stops if  step  size  is
					less than WStep. Recommended value - 0.01.  Zero  step
					size means stopping after MaxIts iterations.
	MaxIts      -   stopping   criterion.  Algorithm  stops  after  MaxIts
					iterations (NOT gradient  calculations).  Zero  MaxIts
					means stopping when step is sufficiently small.

OUTPUT PARAMETERS:
	Network     -   trained neural network.
	Info        -   return code:
					* -8, if both WStep=0 and MaxIts=0
					* -2, if there is a point with class number
						  outside of [0..NOut-1].
					* -1, if wrong parameters specified
						  (NPoints<0, Restarts<1).
					*  2, if task has been solved.
	Rep         -   training report

  -- ALGLIB --
	 Copyright 09.12.2007 by Bochkanov Sergey
*************************************************************************/
func MlpTrainLbfgs(network *mlpbase.Multilayerperceptron, xy *[][]float64, npoints int, decay float64, restarts int, wstep float64, maxits int, info *int, rep *MlpReport) error {
	i := 0
	pass := 0
	nin := 0
	nout := 0
	wcount := 0
	w := make([]float64, 0)
	wbest := make([]float64, 0)
	e := 0.0
	v := 0.0
	ebest := 0.0
	internalrep := &minlbfgsreport{}
	state := NewMinLbfgsState()
	i_ := 0
	*info = 0
	//
	// Test inputs, parse flags, read network geometry
	//
	if wstep == 0 && maxits == 0 {
		*info = -8
		return nil
	}
	if ((npoints <= 0 || restarts < 1) || wstep < 0) || maxits < 0 {
		*info = -1
		return nil
	}
	mlpbase.MlpProperties(network, &nin, &nout, &wcount)
	if mlpbase.MlpIsSoftMax(network) {
		for i = 0; i <= npoints - 1; i++ {
			if utils.RoundInt((*xy)[i][ nin]) < 0 || utils.RoundInt((*xy)[i][ nin]) >= nout {
				*info = -2
				return nil
			}
		}
	}
	decay = math.Max(decay, mindecay)
	*info = 2
	//
	// Prepare
	//
	mlpbase.MlpInitPreprocessor(network, *xy, npoints)
	w = make([]float64, wcount - 1 + 1)
	wbest = make([]float64, wcount - 1 + 1)
	ebest = maxrealnumber
	//
	// Multiple starts
	//
	rep.NCholesky = 0
	rep.NHess = 0
	rep.NGrad = 0
	for pass = 1; pass <= restarts; pass++ {
		//
		// Process
		//
		mlpbase.MlpRandomize(network)
		for i_ = 0; i_ <= wcount - 1; i_++ {
			w[i_] = network.Weights[i_]
		}
		if err := minlbfgscreate(wcount, utils.MinInt(wcount, 10), &w, state); err != nil {
			return err
		}
		if err := minlbfgssetcond(state, 0.0, 0.0, wstep, maxits); err != nil {
			return err
		}
		for minlbfgsiteration(state) {
			for i_ = 0; i_ <= wcount - 1; i_++ {
				network.Weights[i_] = state.x[i_]
			}
			if err := mlpbase.MlpGradNBatch(network, *xy, npoints, &state.f, &state.g); err != nil {
				return err
			}
			v = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				v += network.Weights[i_] * network.Weights[i_]
			}
			state.f = state.f + 0.5 * decay * v
			for i_ = 0; i_ <= wcount - 1; i_++ {
				state.g[i_] = state.g[i_] + decay * network.Weights[i_]
			}
			rep.NGrad += 1
		}
		minlbfgsresults(state, &w, internalrep)
		for i_ = 0; i_ <= wcount - 1; i_++ {
			network.Weights[i_] = w[i_]
		}
		//
		// Compare with best
		//
		v = 0.0
		for i_ = 0; i_ <= wcount - 1; i_++ {
			v += network.Weights[i_] * network.Weights[i_]
		}
		e = mlpbase.MlpErrorN(network, xy, npoints) + 0.5 * decay * v
		if e < ebest {
			for i_ = 0; i_ <= wcount - 1; i_++ {
				wbest[i_] = network.Weights[i_]
			}
			ebest = e
		}
	}
	//
	// The best network
	//
	for i_ = 0; i_ <= wcount - 1; i_++ {
		network.Weights[i_] = wbest[i_]
	}
	return nil
}

/*************************************************************************
Neural network training using early stopping (base algorithm - L-BFGS with
regularization).

INPUT PARAMETERS:
	Network     -   neural network with initialized geometry
	TrnXY       -   training set
	TrnSize     -   training set size
	ValXY       -   validation set
	ValSize     -   validation set size
	Decay       -   weight decay constant, >=0.001
					Decay term 'Decay*||Weights||^2' is added to error
					function.
					If you don't know what Decay to choose, use 0.001.
	Restarts    -   number of restarts from random position, >0.
					If you don't know what Restarts to choose, use 2.

OUTPUT PARAMETERS:
	Network     -   trained neural network.
	Info        -   return code:
					* -2, if there is a point with class number
						  outside of [0..NOut-1].
					* -1, if wrong parameters specified
						  (NPoints<0, Restarts<1, ...).
					*  2, task has been solved, stopping  criterion  met -
						  sufficiently small step size.  Not expected  (we
						  use  EARLY  stopping)  but  possible  and not an
						  error.
					*  6, task has been solved, stopping  criterion  met -
						  increasing of validation set error.
	Rep         -   training report

NOTE:

Algorithm stops if validation set error increases for  a  long  enough  or
step size is small enought  (there  are  task  where  validation  set  may
decrease for eternity). In any case solution returned corresponds  to  the
minimum of validation set error.

  -- ALGLIB --
	 Copyright 10.03.2009 by Bochkanov Sergey
*************************************************************************/
func MlpTraines(network *mlpbase.Multilayerperceptron, trnxy [][]float64, trnsize int, valxy *[][]float64, valsize int, decay float64, restarts int, info *int, rep *MlpReport) {
	i := 0
	pass := 0
	nin := 0
	nout := 0
	wcount := 0
	w := make([]float64, 0)
	wbest := make([]float64, 0)
	e := 0.0
	v := 0.0
	ebest := 0.0
	wfinal := make([]float64, 0)
	efinal := 0.0
	itbest := 0
	internalrep := &minlbfgsreport{}
	state := NewMinLbfgsState()
	wstep := 0.0
	i_ := 0

	*info = 0
	wstep = 0.001

	//
	// Test inputs, parse flags, read network geometry
	//
	if ((trnsize <= 0 || valsize <= 0) || restarts < 1) || decay < 0 {
		*info = -1
		return
	}
	mlpbase.MlpProperties(network, &nin, &nout, &wcount)
	if mlpbase.MlpIsSoftMax(network) {
		for i = 0; i <= trnsize - 1; i++ {
			if utils.RoundInt(trnxy[i][ nin]) < 0 || utils.RoundInt(trnxy[i][nin]) >= nout {
				*info = -2
				return
			}
		}
		for i = 0; i <= valsize - 1; i++ {
			if utils.RoundInt((*valxy)[i][ nin]) < 0 || utils.RoundInt((*valxy)[i][ nin]) >= nout {
				*info = -2
				return
			}
		}
	}
	*info = 2

	//
	// Prepare
	//
	mlpbase.MlpInitPreprocessor(network, trnxy, trnsize)
	w = make([]float64, wcount - 1 + 1)
	wbest = make([]float64, wcount - 1 + 1)
	wfinal = make([]float64, wcount - 1 + 1)
	efinal = maxrealnumber
	for i = 0; i <= wcount - 1; i++ {
		wfinal[i] = 0
	}

	//
	// Multiple starts
	//
	rep.NCholesky = 0
	rep.NHess = 0
	rep.NGrad = 0
	for pass = 1; pass <= restarts; pass++ {
		//
		// Process
		//
		mlpbase.MlpRandomize(network)
		ebest = mlpbase.MlpError(network, valxy, valsize)
		for i_ = 0; i_ <= wcount - 1; i_++ {
			wbest[i_] = network.Weights[i_]
		}
		itbest = 0
		for i_ = 0; i_ <= wcount - 1; i_++ {
			w[i_] = network.Weights[i_]
		}
		minlbfgscreate(wcount, utils.MinInt(wcount, 10), &w, state)
		minlbfgssetcond(state, 0.0, 0.0, wstep, 0)
		minlbfgssetxrep(state, true)
		for minlbfgsiteration(state) {
			//
			// Calculate gradient
			//
			for i_ = 0; i_ <= wcount - 1; i_++ {
				network.Weights[i_] = state.x[i_]
			}
			mlpbase.MlpGradNBatch(network, trnxy, trnsize, &state.f, &state.g)
			v = 0.0
			for i_ = 0; i_ <= wcount - 1; i_++ {
				v += network.Weights[i_] * network.Weights[i_]
			}
			state.f = state.f + 0.5 * decay * v
			for i_ = 0; i_ <= wcount - 1; i_++ {
				state.g[i_] = state.g[i_] + decay * network.Weights[i_]
			}
			rep.NGrad = rep.NGrad + 1

			//
			// Validation set
			//
			if state.xupdated {
				for i_ = 0; i_ <= wcount - 1; i_++ {
					network.Weights[i_] = w[i_]
				}
				e = mlpbase.MlpError(network, valxy, valsize)
				if e < ebest {
					ebest = e
					for i_ = 0; i_ <= wcount - 1; i_++ {
						wbest[i_] = network.Weights[i_]
					}
					itbest = internalrep.iterationscount
				}
				if internalrep.iterationscount > 30 && float64(internalrep.iterationscount) > (1.5 * float64(itbest)) {
					*info = 6
					break
				}
			}
		}
		minlbfgsresults(state, &w, internalrep)

		//
		// Compare with final answer
		//
		if ebest < efinal {
			for i_ = 0; i_ <= wcount - 1; i_++ {
				wfinal[i_] = wbest[i_]
			}
			efinal = ebest
		}
	}

	//
	// The best network
	//
	for i_ = 0; i_ <= wcount - 1; i_++ {
		network.Weights[i_] = wfinal[i_]
	}
}

/*************************************************************************
Subroutine prepares K-fold split of the training set.

NOTES:
	"NClasses>0" means that we have classification task.
	"NClasses<0" means regression task with -NClasses real outputs.
*************************************************************************/
func mlpkfoldsplit(xy *[][]float64, npoints, nclasses, foldscount int, stratifiedsplits bool, folds *[]int) error {
	i := 0
	j := 0
	k := 0

	//
	// test parameters
	//
	if !(npoints > 0) {
		return fmt.Errorf("MLPKFoldSplit: wrong NPoints!")
	}
	if !(nclasses > 1 || nclasses < 0) {
		return fmt.Errorf("MLPKFoldSplit: wrong NClasses!")
	}
	if !(foldscount >= 2 && foldscount <= npoints) {
		return fmt.Errorf("MLPKFoldSplit: wrong FoldsCount!")
	}
	if !(!stratifiedsplits) {
		return fmt.Errorf("MLPKFoldSplit: stratified splits are not supported!")
	}

	//
	// Folds
	//
	*folds = make([]int, npoints - 1 + 1)
	for i = 0; i <= npoints - 1; i++ {
		(*folds)[i] = i * foldscount / npoints
	}
	for i = 0; i <= npoints - 2; i++ {
		j = i + rand.Intn(npoints - i)
		if j != i {
			k = (*folds)[i]
			(*folds)[i] = (*folds)[j]
			(*folds)[j] = k
		}
	}
	return nil
}

/*************************************************************************
Internal cross-validation subroutine
*************************************************************************/
func mlpkfoldcvgeneral(n *mlpbase.Multilayerperceptron, xy *[][]float64, npoints int, decay float64, restarts, foldscount int, lmalgorithm bool, wstep float64, maxits int, info *int, rep *MlpReport, cvrep *MlpCvReport) {
	i := 0
	fold := 0
	j := 0
	k := 0
	network := mlpbase.NewMlp()
	nin := 0
	nout := 0
	rowlen := 0
	wcount := 0
	nclasses := 0
	tssize := 0
	cvssize := 0
	cvset := utils.MakeMatrixFloat64(0, 0)
	testset := utils.MakeMatrixFloat64(0, 0)
	folds := make([]int, 0)
	relcnt := 0
	internalrep := MlpReport{}
	x := make([]float64, 0)
	y := make([]float64, 0)
	i_ := 0

	*info = 0;


	//
	// Read network geometry, test parameters
	//
	mlpbase.MlpProperties(n, &nin, &nout, &wcount)
	if mlpbase.MlpIsSoftMax(n) {
		nclasses = nout
		rowlen = nin + 1
	}else {
		nclasses = -nout
		rowlen = nin + nout
	}
	if (npoints <= 0 || foldscount < 2) || foldscount > npoints {
		*info = -1
		return
	}
	mlpbase.MlpCopy(n, network)

	//
	// K-fold out cross-validation.
	// First, estimate generalization error
	//
	testset = utils.MakeMatrixFloat64(npoints - 1 + 1, rowlen - 1 + 1)
	cvset = utils.MakeMatrixFloat64(npoints - 1 + 1, rowlen - 1 + 1)
	x = make([]float64, nin - 1 + 1)
	y = make([]float64, nout - 1 + 1)
	mlpkfoldsplit(xy, npoints, nclasses, foldscount, false, &folds)
	cvrep.RelclsError = 0
	cvrep.Avgce = 0
	cvrep.RmsError = 0
	cvrep.AvgError = 0
	cvrep.AvgrelError = 0
	rep.NGrad = 0
	rep.NHess = 0
	rep.NCholesky = 0
	relcnt = 0
	for fold = 0; fold <= foldscount - 1; fold++ {
		//
		// Separate set
		//
		tssize = 0
		cvssize = 0
		for i = 0; i <= npoints - 1; i++ {
			if folds[i] == fold {
				for i_ = 0; i_ <= rowlen - 1; i_++ {
					testset[tssize][ i_] = (*xy)[i][ i_]
				}
				tssize = tssize + 1

			}else {
				for i_ = 0; i_ <= rowlen - 1; i_++ {
					cvset[cvssize][ i_] = (*xy)[i][ i_]
				}
				cvssize = cvssize + 1
			}
		}

		//
		// Train on CV training set
		//
		if lmalgorithm {
			MlpTrainLm(network, &cvset, cvssize, decay, restarts, info, &internalrep)
		}else {
			MlpTrainLbfgs(network, &cvset, cvssize, decay, restarts, wstep, maxits, info, &internalrep)
		}
		if *info < 0 {
			cvrep.RelclsError = 0
			cvrep.Avgce = 0
			cvrep.RmsError = 0
			cvrep.AvgError = 0
			cvrep.AvgrelError = 0
			return
		}
		rep.NGrad = rep.NGrad + internalrep.NGrad
		rep.NHess = rep.NHess + internalrep.NHess
		rep.NCholesky = rep.NCholesky + internalrep.NCholesky

		//
		// Estimate error using CV test set
		//
		if mlpbase.MlpIsSoftMax(network) {
			//
			// classification-only code
			//
			cvrep.RelclsError += float64(mlpbase.MlpClsError(network, &testset, tssize))
			cvrep.Avgce += mlpbase.MlpErrorN(network, &testset, tssize)
		}
		for i = 0; i <= tssize - 1; i++ {
			for i_ = 0; i_ <= nin - 1; i_++ {
				x[i_] = testset[i][ i_]
			}
			mlpbase.MlpProcess(network, &x, &y)
			if mlpbase.MlpIsSoftMax(network) {
				//
				// Classification-specific code
				//
				k = utils.RoundInt(testset[i][ nin])
				for j = 0; j <= nout - 1; j++ {
					if j == k {
						cvrep.RmsError = cvrep.RmsError + utils.SqrFloat64(y[j] - 1)
						cvrep.AvgError = cvrep.AvgError + math.Abs(y[j] - 1)
						cvrep.AvgrelError = cvrep.AvgrelError + math.Abs(y[j] - 1)
						relcnt = relcnt + 1
					}else {
						cvrep.RmsError = cvrep.RmsError + utils.SqrFloat64(y[j])
						cvrep.AvgError = cvrep.AvgError + math.Abs(y[j])
					}
				}
			}else {
				//
				// Regression-specific code
				//
				for j = 0; j <= nout - 1; j++ {
					cvrep.RmsError = cvrep.RmsError + utils.SqrFloat64(y[j] - testset[i][ nin + j])
					cvrep.AvgError = cvrep.AvgError + math.Abs(y[j] - testset[i][ nin + j])
					if testset[i][ nin + j] != 0 {
						cvrep.AvgrelError = cvrep.AvgrelError + math.Abs((y[j] - testset[i][ nin + j]) / testset[i][ nin + j])
						relcnt += 1
					}
				}
			}
		}
	}
	if mlpbase.MlpIsSoftMax(network) {
		cvrep.RelclsError = cvrep.RelclsError / float64(npoints)
		cvrep.Avgce = cvrep.Avgce / (math.Log(2) * float64(npoints))
	}
	cvrep.RmsError = math.Sqrt(cvrep.RmsError / float64(npoints * nout))
	cvrep.AvgError = cvrep.AvgError / float64(npoints * nout)
	cvrep.AvgrelError = cvrep.AvgrelError / float64(relcnt)
	*info = 1
}

/*************************************************************************
Cross-validation estimate of generalization error.

Base algorithm - L-BFGS.

INPUT PARAMETERS:
	Network     -   neural network with initialized geometry.   Network is
					not changed during cross-validation -  it is used only
					as a representative of its architecture.
	XY          -   training set.
	SSize       -   training set size
	Decay       -   weight  decay, same as in MLPTrainLBFGS
	Restarts    -   number of restarts, >0.
					restarts are counted for each partition separately, so
					total number of restarts will be Restarts*FoldsCount.
	WStep       -   stopping criterion, same as in MLPTrainLBFGS
	MaxIts      -   stopping criterion, same as in MLPTrainLBFGS
	FoldsCount  -   number of folds in k-fold cross-validation,
					2<=FoldsCount<=SSize.
					recommended value: 10.

OUTPUT PARAMETERS:
	Info        -   return code, same as in MLPTrainLBFGS
	Rep         -   report, same as in MLPTrainLM/MLPTrainLBFGS
	CVRep       -   generalization error estimates

  -- ALGLIB --
	 Copyright 09.12.2007 by Bochkanov Sergey
*************************************************************************/
func Mlpkfoldcvlbfgs(network *mlpbase.Multilayerperceptron, xy *[][]float64, npoints int, decay float64, restarts int, wstep float64, maxits, foldscount int, info *int, rep *MlpReport, cvrep *MlpCvReport) {
	*info = 0
	mlpkfoldcvgeneral(network, xy, npoints, decay, restarts, foldscount, false, wstep, maxits, info, rep, cvrep)
}

/*************************************************************************
Cross-validation estimate of generalization error.

Base algorithm - Levenberg-Marquardt.

INPUT PARAMETERS:
	Network     -   neural network with initialized geometry.   Network is
					not changed during cross-validation -  it is used only
					as a representative of its architecture.
	XY          -   training set.
	SSize       -   training set size
	Decay       -   weight  decay, same as in MLPTrainLBFGS
	Restarts    -   number of restarts, >0.
					restarts are counted for each partition separately, so
					total number of restarts will be Restarts*FoldsCount.
	FoldsCount  -   number of folds in k-fold cross-validation,
					2<=FoldsCount<=SSize.
					recommended value: 10.

OUTPUT PARAMETERS:
	Info        -   return code, same as in MLPTrainLBFGS
	Rep         -   report, same as in MLPTrainLM/MLPTrainLBFGS
	CVRep       -   generalization error estimates

  -- ALGLIB --
	 Copyright 09.12.2007 by Bochkanov Sergey
*************************************************************************/
func Mlpkfoldcvlm(network *mlpbase.Multilayerperceptron, xy *[][]float64, npoints int, decay float64, restarts, foldscount int, info *int, rep *MlpReport, cvrep *MlpCvReport) {
	*info = 0;
	mlpkfoldcvgeneral(network, xy, npoints, decay, restarts, foldscount, true, 0.0, 0, info, rep, cvrep)
}


