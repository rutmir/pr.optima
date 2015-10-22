package entities
import (
	"fmt"
	"time"
	"math"
)

type Efficiency struct {
	TrainType        string     `datastore:"trainType,index" json:"trainType"`
	RangesCount      int32      `datastore:"rangesCount,index" json:"rangesCount"`
	Limit            int32      `datastore:"limit,index" json:"limit"`
	Step             int32      `datastore:"step,index" json:"step"`
	Symbol           string     `datastore:"symbol,index" json:"symbol"`
	SuccessDirection int32      `datastore:"successDirection,index" json:"successDirection"`
	SuccessRange     int32      `datastore:"successRange,index" json:"successRange"`
	LastSD           []int32    `datastore:"lastSD,noindex" json:"lastSD"`
	LastSR           []int32    `datastore:"lastSR,noindex" json:"lastSR"`
	Total            int32      `datastore:"total,index" json:"total"`
	Timestamp        int64      `datastore:"synctime,index" json:"synctime"`
}

func (f *Efficiency) ToString() string {
	return fmt.Sprintf("Score: Symbol: %s\nRanges: %d\nLimit: %d\nStep: %d\nDirectionRate: %v\nRangeRate: %v\nTotal: %d\nLastUpdate: %v",
		f.Symbol,
		f.RangesCount,
		f.Limit,
		f.Step,
		f.GetDirectionRate(),
		f.GetRangeRate(),
		f.Total,
		f.LastUpdate())
}
func (f *Efficiency) GetMlpKey() string {
	return fmt.Sprintf("%d_%s_%d_%d", f.RangesCount, f.TrainType, f.Limit, f.Step)
}
func (f *Efficiency) GetCompositeKey() string {
	return fmt.Sprintf("%s_%s", f.GetMlpKey(), f.Symbol)
}
func (f *Efficiency) GetDirectionRate() float64 {
	if f.Total > 0 {
		return float64(f.SuccessDirection) / float64(f.Total)
	}
	return math.NaN()
}
func (f *Efficiency) GetDirectionRate10() float64 {
	if f.Total > 0 {
		return intSumm(f.LastSD, 10) / 10
	}
	return math.NaN()
}
func (f *Efficiency) GetDirectionRate100() float64 {
	if f.Total > 0 {
		return intSumm(f.LastSD, 100) / 100
	}
	return math.NaN()
}
func (f *Efficiency) GetRangeRate() float64 {
	if f.Total > 0 {
		return float64(f.SuccessRange) / float64(f.Total)
	}
	return math.NaN()
}
func (f *Efficiency) GetRangeRate10() float64 {
	if f.Total > 0 {
		return intSumm(f.LastSR, 10) / 10
	}
	return math.NaN()
}
func (f *Efficiency) GetRangeRate100() float64 {
	if f.Total > 0 {
		return intSumm(f.LastSR, 100) / 100
	}
	return math.NaN()
}
func (f *Efficiency) LastUpdate() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}

func intSumm(a []int32, cnt int) float64 {
	if cnt > 0 {
		if cnt > len(a) {
			return math.NaN()
		}else {
			a = a[len(a) - cnt:]
		}
	}

	result := 0.0
	for item := range a {
		result += float64(item)
	}
	return result
}