package entities
import (
	"time"
	"fmt"
)

type Rate struct {
	Base string     `json:"base"`
	Id   int64      `json:"timestamp"`
	RUB  float32    `json:"RUB"` // Russian Ruble
	JPY  float32    `json:"JPY"` // Yen
	GBP  float32    `json:"GBP"` // Pound Sterling
	USD  float32    `json:"USD"` // US Dollar
	EUR  float32    `json:"EUR"` // Euro
	CNY  float32    `json:"CNY"` // Yuan Renminbi
	CHF  float32    `json:"CHF"` // Swiss Franc
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