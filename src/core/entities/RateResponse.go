package entities

import (
	"fmt"
	"time"
)

// RateResponse struct
type RateResponse struct {
	Base          string             `json:"base"`
	TimestampUnix int64              `json:"timestamp"`
	Disclaimer    string             `json:"disclaimer"`
	License       string             `json:"license"`
	Rates         map[string]float32 `json:"rates"`
}

// ToString method
func (f *RateResponse) ToString() string {
	return fmt.Sprintf("Base: %s\nTimestamp: %v\nRates: %v", f.Base, f.Timestamp(), f.Rates)
}

// ToShortString method
func (f *RateResponse) ToShortString() string {
	return fmt.Sprintf("RateResponse - Base: %s, Timestamp: %v", f.Base, f.Timestamp())
}

// Timestamp method
func (f *RateResponse) Timestamp() time.Time {
	return time.Unix(f.TimestampUnix, 0).UTC()
}

// SetTimestamp method
func (f *RateResponse) SetTimestamp(timestamp time.Time) {
	f.TimestampUnix = timestamp.Unix()
}
