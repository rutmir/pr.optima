package entities
import (
	"fmt"
)

type ResultDataListResponse struct {
	Data     []ResultResponse `json:"data"`
	Score10  float32          `json:"score10"`
	Score100 float32          `json:"score100"`
}

func (f *ResultDataListResponse) ToString() string {
	return fmt.Sprintf("ResultDataListResponse { Data: %s, Data: %v, Score10: %d, Score100: %d.",
		f.Data,
		f.Score10,
		f.Score100)
}
