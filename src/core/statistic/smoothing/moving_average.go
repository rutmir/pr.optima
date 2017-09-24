package smoothing

import "errors"

// SMA - simple moving average
func SMA(source []float64, frame int) ([]float64, error) {
	if source == nil {
		return nil, errors.New("'source' required")
	}
	length := len(source)
	if length < frame || frame < 2 {
		return nil, errors.New("input data is not valid")
	}

	weightedSource := make([]float64, length)
	n := float64(frame)
	for idx, item := range source {
		weightedSource[idx] = item / n
	}

	resultLengs := length - frame + 1
	result := make([]float64, resultLengs)
	var previous float64

	for idx, item := range result {
		if idx == 0 {
			var tmpSum float64
			for i := 0; i < frame; i++ {
				tmpSum += weightedSource[i]
			}
			item = tmpSum
		} else {
			item = previous - weightedSource[idx-1] + weightedSource[idx+frame-1]
		}

		previous = item
	}

	return result, nil
}
