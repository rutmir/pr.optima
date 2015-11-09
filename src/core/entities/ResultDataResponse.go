package entities
import (
	"fmt"
)

type ResultDataResponse struct {
	Data     ResultResponse `json:"data"`
	Score10  float32        `json:"score10"`
	Score100 float32        `json:"score100"`
}

func (f *ResultDataResponse) ToString() string {
	return fmt.Sprintf("ResultDataResponse { Data: %s, Data: %v, Score10: %d, Score100: %d.",
		f.Data.ToString(),
		f.Score10,
		f.Score100)
}
