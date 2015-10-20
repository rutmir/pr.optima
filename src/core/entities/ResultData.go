package entities
import (
	"fmt"
	"time"
)

type ResultData struct {
	Symbol     string     `datastore:"symbol,index" json:"symbol"`
	Timestamp  int64      `datastore:"timestamp,index" json:"timestamp"`
	Step       int32      `datastore:"step,noindex" json:"step"`
	Limit      int32      `datastore:"limit,noindex" json:"limit"`
	Source     []int32    `datastore:"source,noindex" json:"source"`
	Prediction int32      `datastore:"prediction,noindex" json:"prediction"`
}

func (f *ResultData) ToString() string {
	return fmt.Sprintf("ResultData: Symbol: %s\nDatetime: %v\nStep: %d\nLimit: %d\nSource: %v\n\tUSD: %v\nPrediction: %d",
		f.Symbol,
		f.DateCreated(),
		f.Step,
		f.Limit,
		f.Source,
		f.Prediction)
}
func (f *ResultData) DateCreated() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}