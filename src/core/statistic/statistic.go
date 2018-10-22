package statistic

import (
	"errors"
	"math"
	"sort"
)

// CalculateRanges generate diapason of ranges
func CalculateRanges(list []float32, rangesCount int) ([]float64, error) {
	if rangesCount < 1 {
		return nil, errors.New("the ranges count must be positive value more than 1")
	}
	if list == nil || len(list) < rangesCount {
		return nil, errors.New("quantity of set float32 values must be more than the ranges count and more than 1")
	}

	//	var deltas []float32
	var sorted sort.Float64Slice
	var total float64
	for i := 0; i < len(list)-1; i++ {
		delta := float64(list[i+1] / list[i])
		total += delta
		//		deltas = append(deltas, delta)
		sorted = append(sorted, delta)
	}
	sorted.Sort()
	step := total / float64(rangesCount)
	var ranges = make([]float64, rangesCount-1)

	check := false
	for i := 1; i < rangesCount; i++ {
		level := step * float64(i)
		var sum float64
		var prev float64
		for _, element := range sorted {
			sum += element
			if sum > level {
				ranges[i-1] = (element + prev) / 2
				if i > 1 && ranges[i-1] != ranges[i-2] {
					check = true
				}
				break
			} else {
				prev = element
			}
		}
	}

	if check {
		return ranges, nil
	}
	return nil, errors.New("CalculateRanges errer: ranges not vialid")
}

// CalculateEvenRanges generate even diapasons of ranges
func CalculateEvenRanges(list []float32, rangesCount int) ([]float64, error) {
	if rangesCount < 1 || rangesCount%2 == 1 {
		return nil, errors.New("the ranges count must be positive even value more than 1")
	}
	if list == nil || len(list) < rangesCount {
		return nil, errors.New("quantity of set float32 values must be more than the ranges count and more than 1")
	}

	var sortedB sort.Float64Slice
	var sortedL sort.Float64Slice
	var totalB float64
	var totalL float64

	for i := 0; i < len(list)-1; i++ {
		delta := float64(list[i+1] / list[i])
		if delta == 1 {
			continue
		} else {
			if delta > 1 {
				totalB += delta
				sortedB = append(sortedB, delta)
			} else {
				totalL += delta
				sortedL = append(sortedL, delta)
			}
		}
	}
	sortedB.Sort()
	sortedL.Sort()
	half := rangesCount / 2

	// Hight & Low
	rangesB, checkB := evenHalfRanges(half, totalB, sortedB)
	rangesL, checkL := evenHalfRanges(half, totalL, sortedL)

	if checkB && checkL {
		var ranges = make([]float64, rangesCount-1)
		copy(ranges[0:half-1], rangesL)
		ranges[half] = 1.0
		copy(ranges[half+1:rangesCount-2], rangesB)
		return ranges, nil
	}
	return nil, errors.New("CalculateEvenRanges errer: ranges not vialid")
}

// CalculateEvenRanges2 generate even diapasons of ranges
func CalculateEvenRanges2(list []float32, rangesCount int) ([]float64, error) {
	if rangesCount < 4 || rangesCount%2 == 1 {
		return nil, errors.New("the ranges count must be positive even value more than 3")
	}
	if list == nil || len(list) < rangesCount {
		return nil, errors.New("quantity of set float32 values must be more than the ranges count and more than 3")
	}

	var sortedB sort.Float64Slice
	var sortedL sort.Float64Slice
	var totalB float64
	var totalL float64

	for i := 0; i < len(list)-1; i++ {
		delta := float64(list[i+1] / list[i])
		if delta == 1 {
			continue
		} else {
			if delta > 1 {
				totalB += delta - 1
				sortedB = append(sortedB, delta-1)
			} else {
				totalL += 1 - delta
				sortedL = append(sortedL, 1-delta)
			}
		}
	}

	sortedB.Sort()
	sortedL.Sort()
	half := rangesCount / 2
	resultB := evenHalfRanges2(half, totalB, sortedB)
	resultL := evenHalfRanges2(half, totalL, sortedL)

	var ranges = make([]float64, rangesCount-1)

	lenghtL := len(resultL)
	for i := 1; i <= lenghtL; i++ {
		ranges[i-1] = 1 - resultL[lenghtL-i]
	}
	ranges[lenghtL] = 1
	for i := 0; i < len(resultB); i++ {
		ranges[lenghtL+1+i] = 1 + resultB[i]
	}
	return ranges, nil
}

// CalculateClasses convert list of float32 rate values to list int classes, based on diapason ranges
func CalculateClasses(list []float32, ranges []float64) ([]int, error) {
	if list == nil || len(list) < 1 {
		return nil, errors.New("quantity of set float32 (list) values must be more than 1")
	}
	if ranges == nil || len(ranges) < 1 {
		return nil, errors.New("quantity of set float32 (ranges) values must be more than 1")
	}

	deltas := make([]float32, len(list)-1)
	for i := 0; i < len(list)-1; i++ {
		deltas[i] = list[i+1] / list[i]
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

// DetectClass convert float32 value based on diapason ranges to int class
func DetectClass(ranges []float64, element float32) (int, error) {
	if element < 0 {
		return -1, errors.New("the element must be positive value more than 0")
	}
	if len(ranges) == 0 {
		return -1, errors.New("quantity of ranges must be more than 0")
	}

	e := float64(element)

	for i, item := range ranges {
		if e <= item {
			return i, nil
		}
	}

	return len(ranges), nil
}

func evenHalfRanges(half int, total float64, sorted sort.Float64Slice) ([]float64, bool) {
	step := total / float64(half)
	var ranges = make([]float64, half-1)

	check := false
	for i := 1; i < half; i++ {
		level := step * float64(i)
		var sum float64
		var prev float64
		for _, element := range sorted {
			sum += element
			if sum > level {
				ranges[i-1] = (element + prev) / 2
				if i > 1 && ranges[i-1] != ranges[i-2] {
					check = true
				}
				break
			} else {
				prev = element
			}
		}
	}
	return ranges, check
}

func evenHalfRanges2(half int, total float64, sorted sort.Float64Slice) []float64 {
	result := make([]float64, half-1)
	j, sum, prev, step := 0, 0.0, 0.0, total/float64(half)

	for i := 1; i < half; i++ {
		for {
			if j >= len(sorted) {
				break
			}
			element := sorted[j]
			sum += element
			if sum > step*float64(i) {
				if t := int(math.Floor(sum / step)); t > i {
					stepDelta := t - i
					newDelta := (element - prev) / float64(2+stepDelta-1)
					for k := 1; k <= stepDelta; k++ {
						result[i-1] = prev + (newDelta * float64(k))
						i++
					}
				} else {
					result[i-1] = (prev + element) / 2
				}
			}
			prev = element
			j++
		}
	}

	return result
}
