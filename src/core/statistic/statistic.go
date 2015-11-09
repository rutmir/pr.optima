package statistic
import (
	"errors"
	"sort"
	"fmt"
)

func CalculateRanges(list []float32, rangesCount int) ([]float64, error) {
	if rangesCount < 1 {
		return nil, errors.New("the ranges count must be positive value more than 1.")
	}
	if list == nil || len(list) < rangesCount {
		return nil, errors.New("quantity of set float32 values must be more than the ranges count and more than 1.")
	}

	//	var deltas []float32
	var sorted sort.Float64Slice
	var total float64 = 0
	for i := 0; i < len(list) - 1; i++ {
		delta := float64(list[i + 1] / list[i])
		total += delta
		//		deltas = append(deltas, delta)
		sorted = append(sorted, delta)
	}
	sorted.Sort()
	step := total / float64(rangesCount)
	var ranges = make([]float64, rangesCount - 1)

	check := false
	for i := 1; i < rangesCount; i++ {
		level := step * float64(i)
		var sum float64 = 0
		var prev float64 = 0
		for _, element := range sorted {
			sum += element
			if sum > level {
				ranges[i - 1] = (element + prev) / 2
				if i > 1 && ranges[i - 1] != ranges[i - 2] {
					check = true
				}
				break
			}else {
				prev = element
			}
		}
	}

	if check {
		return ranges, nil
	}
	return nil, fmt.Errorf("CalculateRanges errer: ranges not vialid.")
}

func CalculateClasses(list []float32, ranges []float64) ([]int, error) {
	if list == nil || len(list) < 1 {
		return nil, errors.New("quantity of set float32 (list) values must be more than 1.")
	}
	if ranges == nil || len(ranges) < 1 {
		return nil, errors.New("quantity of set float32 (ranges) values must be more than 1.")
	}

	deltas := make([]float32, len(list) - 1)
	for i := 0; i < len(list) - 1; i++ {
		deltas[i] = list[i + 1] / list[i]
	}

	var classes = make([]int, len(deltas))
	for i, element := range deltas {
		class, err := DetectClass(ranges, element)
		if err != nil {
			return nil, err
		}
		classes[i] = class
	}
	return classes, nil
}

func DetectClass(ranges []float64, element float32) (int, error) {
	if element < 0 {
		return -1, errors.New("the element must be positive value more than 0.")
	}
	if ranges == nil || len(ranges) == 0 {
		return -1, errors.New("quantity of ranges must be more than 0.")
	}

	e := float64(element)

	for i, item := range ranges {
		if e <= item {
			return i, nil
		}
	}

	return len(ranges), nil
}