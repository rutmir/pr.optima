package statistic

import (
	"errors"
	"sort"
)

func CalculateRanges(list []float32, rangesCount int) ([]float32, error) {
	if rangesCount < 1 {
		return nil, errors.New("the ranges count must be positive value more than 1.")
	}
	if list == nil || len(list) < 1 || len(list) < rangesCount {
		return nil, errors.New("quantity of set float32 values must be more than the ranges count and more than 1.")
	}

	var deltas []float32
	var sorted sort.Float64Slice
	var total float32 = 0
	for i := 0; i < len(list) - 1; i++ {
		delta := list[i + 1] / list[i]
		total += delta
		deltas = append(deltas, delta)
		sorted = append(sorted, float64(delta))
	}
	sorted.Sort()
	step := total / float32(rangesCount)
	var ranges []float32

	for i := 1; i < rangesCount; i++ {
		level := float64(step * float32(i))
		var sum float64 = 0
		var prev float64 = 0
		for _, element := range sorted {
			sum += element
			if sum > level {
				ranges = append(ranges, float32(element + prev) / 2)
				break
			}else {
				prev = element
			}
		}
	}

	return ranges, nil
}

func CalculateClasses(list []float32, ranges []float32) ([]int, error) {
	if list == nil || len(list) < 1 {
		return nil, errors.New("quantity of set float32 (list) values must be more than 1.")
	}
	if ranges == nil || len(ranges) < 1 {
		return nil, errors.New("quantity of set float32 (ranges) values must be more than 1.")
	}

	var deltas []float32
	for i := 0; i < len(list) - 1; i++ {
		deltas = append(deltas, list[i + 1] / list[i])
	}

	var classes []int
	for _, element := range deltas {
		class, err := DetectClass(ranges, element)
		if err != nil {
			return nil, err
		}
		classes = append(classes, class)
	}

	return classes, nil
}

func DetectClass(ranges []float32, element float32) (int, error) {
	if element < 0 {
		return -1, errors.New("the element must be positive value more than 0.")
	}
	if ranges == nil || len(ranges) == 0 {
		return -1, errors.New("quantity of ranges must be more than 0.")
	}

	if (element < ranges[0]) {
		return 0, nil
	}

	for i, line := range ranges {
		if element < line {
			return i, nil
		}
	}

	return len(ranges), nil
}