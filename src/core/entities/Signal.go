package entities

import (
	"fmt"
	"time"
)

// Signal struct
type Signal struct {
	Score10     float32 `json:"score10"`
	Score100    float32 `json:"score100"`
	Timestamp   int64   `json:"timestamp"`
	RangesCount int32   `json:"rangesCount"`
	Prediction  int32   `json:"prediction"`
	Symbol      string  `json:"symbol"`
}

// ToString method
func (f *Signal) ToString() string {
	return fmt.Sprintf("Signal { Symbol: %s; Ranges: %d; Datetime: %v; Prediction: %d; Score10: %v; Score100: %v }",
		f.Symbol,
		f.RangesCount,
		f.DateCreated(),
		f.Prediction,
		f.Score10,
		f.Score100)
}

// DateCreated method
func (f *Signal) DateCreated() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}
