package entities

import (
	"fmt"
	"math"
	"time"
)

// Efficiency struct
type Efficiency struct {
	TrainType   string  `datastore:"trainType,index" json:"trainType"`
	RangesCount int32   `datastore:"rangesCount,index" json:"rangesCount"`
	Limit       int32   `datastore:"limit,index" json:"limit"`
	Frame       int32   `datastore:"frame,index" json:"frame"`
	Symbol      string  `datastore:"symbol,index" json:"symbol"`
	LastSD      []int32 `datastore:"lastSD,noindex" json:"lastSD"`
	Timestamp   int64   `datastore:"timestamp,index" json:"timestamp"`
}

// ToString method
func (f *Efficiency) ToString() string {
	return fmt.Sprintf("Efficiency {: Symbol: %s, Ranges: %d, Limit: %d, Frame: %d, DirectionRate10: %v, DirectionRate100: %v, Timestamp: %v }",
		f.Symbol,
		f.RangesCount,
		f.Limit,
		f.Frame,
		f.GetDirectionRate10(),
		f.GetDirectionRate100(),
		f.LastUpdate())
}

// GetMlpKey method
func (f *Efficiency) GetMlpKey() string {
	return fmt.Sprintf("%d_%s_%d_%d", f.RangesCount, f.TrainType, f.Limit, f.Frame)
}

// GetCompositeKey method
func (f *Efficiency) GetCompositeKey() string {
	return fmt.Sprintf("%s_%s", f.GetMlpKey(), f.Symbol)
}

// GetDirectionRate10 method
func (f *Efficiency) GetDirectionRate10() float64 {
	if f.LastSD != nil && len(f.LastSD) >= 10 {
		return intSumm(f.LastSD, 10) / 10
	}
	return math.NaN()
}

// GetDirectionRate100 method
func (f *Efficiency) GetDirectionRate100() float64 {
	if f.LastSD != nil && len(f.LastSD) >= 100 {
		return intSumm(f.LastSD, 100) / 100
	}
	return math.NaN()
}

// LastUpdate method
func (f *Efficiency) LastUpdate() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}

func intSumm(a []int32, cnt int) float64 {
	if cnt <= 0 || cnt > len(a) {
		return math.NaN()
	}

	var result int32
	for i := len(a) - 1; i > -1 && cnt > 0; i-- {
		result += a[i]
	}
	return float64(result)
}
