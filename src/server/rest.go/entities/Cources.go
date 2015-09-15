package entities
import (
	"fmt"
	"time"
)

type Cources struct {
	Id         		int        			`json:"id"`
	Base       		string        		`json:"base"`
	TimestampUnix  	int64        		`json:"timestamp"`
	Disclaimer 		string        		`json"disclaimer"`
	License			string				`json"license"`
	Rates      		map[string]float32  `json"rates"`
}

func (f Cources) ToString() string {
	return fmt.Sprintf("Base: %s\nTimestamp: %v\nRates: %v", f.Base, f.Timestamp(), f.Rates)
}
func (f Cources) Timestamp() time.Time {
	return time.Unix(f.TimestampUnix, 0).UTC()
}
func (f Cources) SetTimestamp(timestamp time.Time) {
	f.TimestampUnix = timestamp.Unix()
}