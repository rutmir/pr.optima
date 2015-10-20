package utils

import (
	"fmt"
	"math"
)

func MakeMatrixFloat64(n, m int) [][]float64 {
	result := make([][]float64, n)
	for i := range result {
		result[i] = MakeSliceFloat64(m)
	}
	return result
}

func MakeSliceFloat64(n int) []float64 {
	return make([]float64, n)
}

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func SignInt(value float64) int {
	if value > 0 {
		return 1
	}
	if value < 0 {
		return -1
	}
	return 0
}

func SignFloat64(value float64) float64 {
	if value > 0 {
		return 1.0
	}
	if value < 0 {
		return -1.0
	}
	return 0.0
}

/*************************************************************************
This function checks that all values from X[] are finite

  -- ALGLIB --
	 Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
func IsFiniteVector(x []float64, n int) (bool, error) {
	if !(n >= 0) {
		return false, fmt.Errorf("APSERVIsFiniteVector: internal error (N<0)")
	}

	for i := 0; i <= n - 1; i++ {
		if !IsFinite(x[i]) {
			return false, nil
		}
	}

	return true, nil
}

func Round(f float64) float64 {
	return math.Floor(f + .5)
}

func RoundInt(f float64) int {
	return int(math.Floor(f + .5))
}

func IsFinite(d float64) bool {
	return !math.IsNaN(d) && !(d > math.MaxFloat64 || d < -math.MaxFloat64)
}

func AbsComplex(x float64, y float64) float64 {
	xabs := math.Abs(x);
	yabs := math.Abs(y);
	w := 0.0
	if xabs > yabs {
		w = xabs
	}else {
		w = yabs
	}
	v := 0.0
	if xabs < yabs {
		v = xabs
	}else {
		v = yabs
	}
	if v == 0 {
		return w
	}else {
		{
			t := v / w
			return w * math.Sqrt(1 + t * t)
		}
	}
}

func CloneArrayInt(src []int) []int {
	result := make([]int, len(src))
	copy(result, src)
	return result
}

func CloneArrayFloat64(src []float64) []float64 {
	result := make([]float64, len(src))
	copy(result, src)
	return result
}

func CloneMatrixFloat64(src [][]float64) [][]float64 {
	l := len(src)
	result := make([][]float64, l)

	for i := 0; i < l; i++ {
		result[i] = CloneArrayFloat64(src[i])
	}

	return result
}

func SqrFloat64(a float64) float64 {
	return a * a
}

func SqrInt(a int) int {
	return a * a
}