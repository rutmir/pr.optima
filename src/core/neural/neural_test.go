package neural_test
import (
	"testing"
	"fmt"
	"pr.optima/src/core/neural"
)


func TestNeuron(t *testing.T) {
	limit := 20
	step := 5

	data := [...]int{4, 4, 0, 0, 0, 0, 1, 4, 0, 1, 0, 0, 0, 0, 5, 5, 1, 4, 0, 5, 0, 0, 0, 0, 1, 1, 2, 4, 4, 1, 2, 2, 4, 2, 2, 4, 1, 5, 1, 0, 0, 4, 5, 5, 4, 4, 5, 1, 4, 1, 4, 4, 1, 0, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 1, 5, 0, 2, 5, 5, 5, 0, 0, 5, 0, 4, 1, 5, 1, 5, 4, 4, 5, 5, 5, 5, 1, 1, 4, 1, 1, 2, 1, 4, 1, 1, 4, 5, 4, 5, 5, 4, 5, 1, 1, 0, 5, 0, 5 }
	t.Log(len(data))
	dataF := make([]float64, len(data))
	for i, item := range data {
		dataF[i] = float64(item) / 5.0
	}
	cnt := ((len(dataF) - limit) / step) + 1
	train := make([][]float64, cnt)

	for i := 0; i < cnt; i++ {
		train[i] = make([]float64, limit)
		for j := 0; j < limit; j++ {
			train[i][j] = dataF[step * i + j]
		}
	}
	process := dataF[len(dataF) - step:]

	// create
	mlp := neural.MlpCreate1(step, step, 1)

	// teach


	// get result
	result := neural.MlpProcess(mlp, &process)

	fmt.Printf("nueral result: %v\n", result)
}