package entities
import (
	"fmt"
	"time"
)

type ResultData struct {
	RangesCount int32      `datastore:"rangesCount,noindex" json:"rangesCount"`
	Limit       int32      `datastore:"limit,noindex" json:"limit"`
	TrainType   string     `datastore:"trainType,noindex" json:"trainType"`
	Step        int32      `datastore:"step,noindex" json:"step"`
	Symbol      string     `datastore:"symbol,index" json:"symbol"`
	Timestamp   int64      `datastore:"timestamp,index" json:"timestamp"`
	Source      []int32    `datastore:"source,noindex" json:"source"`
	Prediction  int32      `datastore:"prediction,noindex" json:"prediction"`
	Result      int32      `datastore:"result,noindex" json:"result"`
}

func (f *ResultData) ToString() string {
	return fmt.Sprintf("ResultData: Symbol: %s\nRanges: %d\nLimit: %d\nStep: %d\nDatetime: %v\nSource: %v\nPrediction: %d\nResult: %d.",
		f.Symbol,
		f.RangesCount,
		f.Limit,
		f.Step,
		f.DateCreated(),
		f.Source,
		f.Prediction,
		f.Result)
}
func (f *ResultData) DateCreated() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}
func (f *ResultData) GetMlpKey() string {
	return fmt.Sprintf("%d_%s_%d_%d", f.RangesCount, f.TrainType, f.Limit, f.Step)
}
func (f *ResultData) GetMlpSymbolKey() string {
	return fmt.Sprintf("%s_%s", f.GetMlpKey(), f.Symbol)
}
func (f *ResultData) GetCompositeKey() string {
	return fmt.Sprintf("%s_%d", f.GetMlpSymbolKey(), f.Timestamp)
}