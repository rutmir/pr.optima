package responses
import (
	"fmt"
	"time"
)

type RateResponse struct {
	Base       		string        		`json:"base"`
	TimestampUnix  	int64        		`json:"timestamp"`
	Disclaimer string                `json:"disclaimer"`
	License    string                `json:"license"`
	Rates      map[string]float32  `json:"rates"`
}

func (f RateResponse) ToString() string {
	return fmt.Sprintf("Base: %s\nTimestamp: %v\nRates: %v", f.Base, f.Timestamp(), f.Rates)
}
func (f RateResponse) Timestamp() time.Time {
	return time.Unix(f.TimestampUnix, 0).UTC()
}
func (f RateResponse) SetTimestamp(timestamp time.Time) {
	f.TimestampUnix = timestamp.Unix()
}