package entities
import (
	"fmt"
	"time"
)

type Signal struct {
	RangesCount int32      `json:"rangesCount"`
	Symbol      string     `json:"symbol"`
	Timestamp   int64      `json:"timestamp"`
	Prediction  int32      `json:"prediction"`
	Score10     float32    `json:"score10"`
	Score100    float32    `json:"score100"`
}

func (f *Signal) ToString() string {
	return fmt.Sprintf("Signal { Symbol: %s; Ranges: %d; Datetime: %v; Prediction: %d; Score10: %v; Score100: %v }",
		f.Symbol,
		f.RangesCount,
		f.DateCreated(),
		f.Prediction,
		f.Score10,
		f.Score100)
}
func (f *Signal) DateCreated() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}