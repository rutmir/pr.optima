package entities
import (
	"fmt"
	"time"
)

type Efficiency struct {
	TrainType     string     `datastore:"trainType,noindex" json:"trainType"`
	RangesCount   int32      `datastore:"rangesCount,noindex" json:"rangesCount"`
	Limit         int32      `datastore:"limit,noindex" json:"limit"`
	Step          int32      `datastore:"step,noindex" json:"step"`
	Symbol        string     `datastore:"symbol,index" json:"symbol"`
	DirectionRate float32    `datastore:"directionRate,noindex" json:"directionRate"`
	RangeRate     float32    `datastore:"rangeRate,noindex" json:"rangeRate"`
}

func (f *Efficiency) ToString() string {
	return fmt.Sprintf("Score: Symbol: %s\nRanges: %d\nLimit: %d\nStep: %d\nDirectionRate: %v\nRangeRate: %v",
		f.Symbol,
		f.RangesCount,
		f.Limit,
		f.Step,
		f.DirectionRate,
		f.RangeRate)
}
func (f *Efficiency) GetMlpKey() string {
	return fmt.Sprintf("%d_%s_%d_%d", f.RangesCount, f.TrainType, f.Limit, f.Step)
}
func (f *Efficiency) GetCompositeKey() string {
	return fmt.Sprintf("%d_%s_%d_%d_%s", f.RangesCount, f.TrainType, f.Limit, f.Step, f.Symbol)
}