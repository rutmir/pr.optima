package entities
import (
	"fmt"
	"time"
)

type ResultResponse struct {
	Symbol     string     `json:"symbol"`
	Timestamp  int64      `json:"timestamp"`
	Data       []float32  `json:"data"`
	Source     []int32    `json:"source"`
	Time       []int64    `json:"time"`
	Prediction int32      `json:"prediction"`
	Result     int32      `json:"result"`
	Levels     int32      `json:"levels"`
}

func (f *ResultResponse) ToString() string {
	return fmt.Sprintf("ResultResponse { Symbol: %s, Timestamp: %v, Data: %v, Source: %v, Prediction: %d, Result: %d, Levels: %d.",
		f.Symbol,
		f.DateCreated(),
		f.Data,
		f.Source,
		f.Prediction,
		f.Result,
		f.Levels)
}
func (f *ResultResponse) DateCreated() time.Time {
	return time.Unix(f.Timestamp, 0).UTC()
}
func (f *ResultResponse) Clone() *ResultResponse {
	result := new(ResultResponse)
	result.Symbol = f.Symbol
	result.Timestamp = f.Timestamp
	result.Prediction = f.Prediction
	result.Result = f.Result
	result.Levels = f.Levels

	result.Data = make([]float32, len(f.Data))
	copy(result.Data, f.Data)
	result.Source = make([]int32, len(f.Source))
	copy(result.Source, f.Source)
	result.Time = make([]int64, len(f.Time))
	copy(result.Time, f.Time)

	return result
}