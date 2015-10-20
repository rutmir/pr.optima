package work
import (
	"pr.optima/src/core/neural"
	"pr.optima/src/core/entities"
	"pr.optima/src/core/statistic"
	"fmt"
	"math"
	"pr.optima/src/repository"
)

type Work struct {
	mlp        *neural.MultiLayerPerceptron
	frame      int
	step       int
	rangeCount int
	hIn        int
	symbol     string

	loopCount  int
	ranges     []float64
	repo       repository.ResultDataRepo
}

func NewWork(rCount, step, limit, hIn int, symbol string) *Work {
	result := new(Work)
	result.symbol = symbol
	result.frame = limit
	result.step = step
	result.rangeCount = rCount
	result.hIn = hIn
	result.mlp = neural.MlpCreate1(step, step, hIn)

	result.loopCount = 0
	result.ranges = nil
	result.repo = repository.NewResultDataRepo(limit, true, symbol)
	fmt.Printf("NewResultDataRepo length %v\n", result.repo.Len())

	return result
}

func (f *Work)Process(rates []entities.Rate) (int, error) {
	var rawSource []entities.Rate
	if len(rates) > f.frame + 1 {
		rawSource = rates[len(rates) - f.frame - 1:]
	}

	_time := rawSource[len(rawSource) - 1].Id
	source := extractFloatSet(rawSource, f.symbol)

	if (f.loopCount > f.step || f.ranges == nil) {
		var err error
		if f.ranges, err = statistic.CalculateRanges(source, f.rangeCount); err != nil {
			f.ranges = nil
			return -1, err
		}

		// re train mlp
		classes, err := statistic.CalculateClasses(source, f.ranges)
		if err != nil {
			return -1, err
		}

		train := make([][]float64, 1)
		train[0] = convertArrayToFloat64(classes)

		info, _, err := neural.MlpTrainLbfgs(f.mlp, &train, 1, 0.001, 2, 0.01, 0)
		if err != nil {
			return -1, err
		}else if info != 2 {
			return -1, fmt.Errorf("MlpTrainLbfgs error info param: %d.", info)
		}
		f.loopCount = 0
	}

	if f.ranges != nil {
		f.loopCount++
		source, err := statistic.CalculateClasses(source[len(source) - f.step - 1:], f.ranges)
		if err != nil {
			return -1, err
		}
		// process
		process := convertArrayToFloat64(source)
		rawResult := neural.MlpProcess(f.mlp, &process)

		result := entities.ResultData{
			Symbol:      f.symbol,
			Timestamp:   _time,
			Step:        int32(f.step),
			Limit:       int32(f.frame),
			Prediction:  int32(math.Floor((*rawResult)[0] + .5)),
			Source:      convertArrayToInt32(source)}

		if err := f.repo.Push(result); err != nil {
			return -1, err
		}
		if rawResult != nil || len(*rawResult) > 0 {
			return int(result.Prediction), nil
		}
	}

	return -1, nil
}

func extractFloatSet(rates []entities.Rate, symbol string) []float32 {
	result := make([]float32, len(rates))

	switch symbol {
	case "CHF":
		for i, element := range rates { result[i] = element.CHF }
	case "CNY":
		for i, element := range rates { result[i] = element.CNY }
	case "EUR":
		for i, element := range rates { result[i] = element.EUR }
	case "GBP":
		for i, element := range rates { result[i] = element.GBP }
	case "JPY":
		for i, element := range rates { result[i] = element.JPY }
	case "RUB":
		for i, element := range rates { result[i] = element.RUB }
	case "USD":
		for i, element := range rates { result[i] = element.USD }
	}

	return result
}

func convertArrayToFloat64(a []int) []float64 {
	result := make([]float64, len(a))
	for i, item := range a {
		result[i] = float64(item)
	}
	return result
}

func convertArrayToInt32(a []int) []int32 {
	result := make([]int32, len(a))
	for i, item := range a {
		result[i] = int32(item)
	}
	return result
}