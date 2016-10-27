package entities

import (
	"fmt"
	"time"
)

// Rate2Response struct
type Rate2Response struct {
	Success       bool               `json:"success"`
	Terms         string             `json:"terms"`
	Privacy       string             `json:"privacy"`
	TimestampUnix int64              `json:"timestamp"`
	Base          string             `json:"source"`
	Quotes        map[string]float32 `json:"quotes"`
}

// ToString method
func (f Rate2Response) ToString() string {
	return fmt.Sprintf("Base: %s\nTimestamp: %v\nRates: %v", f.Base, f.Timestamp(), f.Quotes)
}

// ToShortString method
func (f Rate2Response) ToShortString() string {
	return fmt.Sprintf("Rate2Response - Base: %s, Timestamp: %v", f.Base, f.Timestamp())
}

// Timestamp method
func (f Rate2Response) Timestamp() time.Time {
	return time.Unix(f.TimestampUnix, 0).UTC()
}

// SetTimestamp method
func (f Rate2Response) SetTimestamp(timestamp time.Time) {
	f.TimestampUnix = timestamp.Unix()
}
