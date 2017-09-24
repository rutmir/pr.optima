package jobs

import (
	"errors"
	"fmt"
	"math"
	"net/http"

	"pr.optima/src/core/entities"
	"pr.optima/src/core/neural"
	"pr.optima/src/core/statistic"
	"pr.optima/src/repository"
)

const (
	// TTLbfgs type of training neurones
	TTLbfgs = "L-BFGS"
)

type fetchRatesWorkItem struct {
	Limit      int
	mlp        *neural.MultiLayerPerceptron
	frame      int
	rangeCount int
	hIn        int
	symbol     string
	trainType  string
	loopCount  int
	ranges     []float64
}

func newFetchRatesWorkItem(rCount, frame, limit, hIn int, trainType, symbol string) *fetchRatesWorkItem {
	result := new(fetchRatesWorkItem)
	result.symbol = symbol
	result.Limit = limit
	result.frame = frame
	result.rangeCount = rCount
	result.trainType = trainType
	result.hIn = hIn
	result.mlp = neural.MlpCreate1(frame, frame, hIn)
	result.loopCount = 0
	result.ranges = nil

	return result
}

func (f *fetchRatesWorkItem) Process(rates []entities.Rate, r *http.Request) (int, error) {
	// prepare income data
	var rawSource []entities.Rate = rates
	if len(rates) > f.Limit+1 {
		rawSource = rates[len(rates)-f.Limit-1:]
	}

	_time := rawSource[len(rawSource)-1].ID
	source, isValid := extractFloatSet(rawSource, f.symbol)
	sourceLength := len(source)

	// assess previous prediction
	if f.ranges != nil && len(f.ranges) > 0 && sourceLength > 1 {
		class, err := statistic.DetectClass(f.ranges, source[sourceLength-1]/source[sourceLength-2])
		if err != nil {
			return -1, err
		}
		resultRepo := repository.NewResultDataRepo(f.Limit, true, f.symbol, r)
		effRepo := repository.NewEfficiencyRepo(f.trainType, f.symbol, int32(f.rangeCount), int32(f.Limit), int32(f.frame), nil)
		if last, found := resultRepo.Get(rawSource[sourceLength-2].ID); found {
			eff, _ := effRepo.GetLast()
			last.Result = int32(class)
			if last.Prediction == last.Result {
				// prediction and result completely match
				eff.LastSD = append(eff.LastSD, 1)
			} else {
				rcHalf := float32(f.rangeCount-1) / 2
				fClass := float32(class)
				fPrediction := float32(last.Prediction)
				if (rcHalf < fClass && rcHalf < fPrediction) || (rcHalf > fClass && rcHalf > fPrediction) {
					eff.LastSD = append(eff.LastSD, 1)
				} else {
					eff.LastSD = append(eff.LastSD, 0)
				}
			}
			if len(eff.LastSD) > 100 {
				eff.LastSD = eff.LastSD[len(eff.LastSD)-100:]
			}

			eff.Timestamp = last.Timestamp

			if err := resultRepo.Sync(last); err != nil {
				return -1, err
			}
			if err := effRepo.Sync(eff); err != nil {
				return -1, err
			}
		}
	}

	if !isValid {
		f.ranges = nil
		return -1, errors.New("no activity detected")
	}

	// retrain mlp
	if f.loopCount > f.frame || f.ranges == nil {
		var err error
		if f.ranges, err = statistic.CalculateEvenRanges2(source, f.rangeCount); err != nil {
			f.ranges = nil
			return -1, err
		}

		// retrain mlp
		classes, err := statistic.CalculateClasses(source, f.ranges)
		if err != nil {
			return -1, err
		}

		train := make([][]float64, 1)
		train[0] = convertArrayToFloat64(classes)

		switch f.trainType {
		case TTLbfgs:
			info, _, err := neural.MlpTrainLbfgs(f.mlp, &train, 1, 0.001, 2, 0.01, 0)
			if err != nil {
				return -1, err
			} else if info != 2 {
				return -1, fmt.Errorf("MlpTrainLbfgs error info param: %d.", info)
			}
		default:
			return -1, errors.New("unknowen trainig type")
		}

		f.loopCount = 0
	}

	if f.ranges != nil {
		f.loopCount++
		source, err := statistic.CalculateClasses(source[len(source)-f.frame-1:], f.ranges)
		if err != nil {
			return -1, err
		}
		// process
		process := convertArrayToFloat64(source)
		rawResult := neural.MlpProcess(f.mlp, &process)
		result := entities.ResultData{
			RangesCount: int32(f.rangeCount),
			TrainType:   f.trainType,
			Limit:       int32(f.Limit),
			Step:        int32(f.frame),
			Symbol:      f.symbol,
			Timestamp:   _time,
			Source:      convertArrayToInt32(source),
			Prediction:  int32(math.Floor((*rawResult)[0] + .5)),
			Result:      -1}
		resultRepo := repository.NewResultDataRepo(f.Limit, true, f.symbol, r)
		if err := resultRepo.Push(result); err != nil {
			return -1, err
		}
		if rawResult != nil || len(*rawResult) > 0 {
			return int(result.Prediction), nil
		}
	}
	return -1, nil
}

func extractFloatSet(rates []entities.Rate, symbol string) ([]float32, bool) {
	l := len(rates)
	result := make([]float32, l)

	switch symbol {
	case "CHF":
		for i, element := range rates {
			result[i] = element.CHF
		}
	case "CNY":
		for i, element := range rates {
			result[i] = element.CNY
		}
	case "EUR":
		for i, element := range rates {
			result[i] = element.EUR
		}
	case "GBP":
		for i, element := range rates {
			result[i] = element.GBP
		}
	case "JPY":
		for i, element := range rates {
			result[i] = element.JPY
		}
	case "RUB":
		for i, element := range rates {
			result[i] = element.RUB
		}
	case "USD":
		for i, element := range rates {
			result[i] = element.USD
		}
	}

	if l < 2 {
		return result, true
	}

	cnt := 3
	if l < cnt {
		cnt = l
	}

	isValid := false
	for i := 1; i < cnt; i++ {
		if result[l-i] != result[l-i-1] {
			isValid = true
			break
		}
	}
	return result, isValid
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
