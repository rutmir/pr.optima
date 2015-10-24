package entities
import (
	"fmt"
	"time"
)

type Rate2Response struct {
	Success       bool `json:"success"`
	Terms         string              `json:"terms"`
	Privacy       string              `json:"privacy"`
	TimestampUnix int64               `json:"timestamp"`
	Base          string              `json:"source"`
	Quotes        map[string]float32  `json:"quotes"`
}

func (f Rate2Response) ToString() string {
	return fmt.Sprintf("Base: %s\nTimestamp: %v\nRates: %v", f.Base, f.Timestamp(), f.Quotes)
}
func (f Rate2Response) ToShortString() string {
	return fmt.Sprintf("Rate2Response - Base: %s, Timestamp: %v", f.Base, f.Timestamp())
}
func (f Rate2Response) Timestamp() time.Time {
	return time.Unix(f.TimestampUnix, 0).UTC()
}
func (f Rate2Response) SetTimestamp(timestamp time.Time) {
	f.TimestampUnix = timestamp.Unix()
}