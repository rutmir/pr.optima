package entities

import (
	"fmt"
)

// ResultDataResponse struct
type ResultDataResponse struct {
	Data     ResultResponse `json:"data"`
	Score10  float32        `json:"score10"`
	Score100 float32        `json:"score100"`
}

// ToString method
func (f *ResultDataResponse) ToString() string {
	return fmt.Sprintf("ResultDataResponse { Data: %v, Score10: %v, Score100: %v.",
		f.Data.ToString(),
		f.Score10,
		f.Score100)
}
