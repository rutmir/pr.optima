package entities
import (
	"time"
	"fmt"
)

type Rate struct {
	Base string                `json:"base"`
	Id   int64                `json:"timestamp"`
	RUB  float32 // Russian Ruble
	JPY  float32 // Yen
	GBP  float32 // Pound Sterling
	USD  float32 // US Dollar
	EUR  float32 // Euro
	CNY  float32 // Yuan Renminbi
	CHF  float32 // Swiss Franc
}

func (f Rate) ToString() string {
	return fmt.Sprintf("Rate: Base: %s\nDatetime: %v\n\tRUB: %v\n\tJPY: %v\n\tGBP: %v\n\tUSD: %v\n\tEUR: %v\n\tCHY: %v\n\tCHF: %v",
		f.Base,
		f.Timestamp(),
		f.RUB,
		f.JPY,
		f.GBP,
		f.USD,
		f.EUR,
		f.CNY,
		f.CHF)
}
func (f Rate) Timestamp() time.Time {
	return time.Unix(f.Id, 0).UTC()
}
func (f Rate) SetTimestamp(timestamp time.Time) {
	f.Id = timestamp.Unix()
}