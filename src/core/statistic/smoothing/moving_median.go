package smoothing

import "errors"

// MM - moving median
func MM(source []float64, q int) ([]float64, error) {
	if source == nil {
		return nil, errors.New("'source' required")
	}
	length := len(source)
	if length < q*2 || length < 3 || q < 1 {
		return nil, errors.New("input data is not valid")
	}

	weightedSource := make([]float64, length)
	n := float64(q * 2)
	for idx, item := range source {
		weightedSource[idx] = item / n
	}

	result := make([]float64, length)
	var previous float64

	for idx, item := range result {
		if idx < q {
			item = (source[idx] + source[idx+1] + ((3 * source[idx+1]) - (2 * source[idx+2]))) / 3
			previous = item
			continue
		}

		if idx > length-q {
			item = (source[idx] + source[idx-1] + ((3 * source[idx-1]) - (2 * source[idx-2]))) / 3
			previous = item
			continue
		}

		if idx == q {
			var tmpSum float64
			for i := 0; i < n; i++ {
				tmpSum += weightedSource[i]
			}
			item = tmpSum
			previous = item
			continue
		}

		item = previous - weightedSource[idx-1] + weightedSource[idx+q-1]
		previous = item
	}

	return result, nil
}
