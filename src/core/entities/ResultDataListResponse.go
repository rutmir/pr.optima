package entities

import (
	"fmt"
)

// ResultDataListResponse struct
type ResultDataListResponse struct {
	Data     []ResultResponse `json:"data"`
	Score10  float32          `json:"score10"`
	Score100 float32          `json:"score100"`
}

// ToString method
func (f *ResultDataListResponse) ToString() string {
	return fmt.Sprintf("ResultDataListResponse { Data: %v, Score10: %v, Score100: %v.",
		f.Data,
		f.Score10,
		f.Score100)
}
