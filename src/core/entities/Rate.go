package entities
import (
	"time"
	"fmt"
)

type Rate struct {
	Base string     `datastore:"base,noindex" json:"base"`
	Id   int64      `datastore:"id,index" json:"timestamp"`
	RUB  float32    `datastore:"rub,noindex" json:"RUB"` // Russian Ruble
	JPY  float32    `datastore:"jpy,noindex" json:"JPY"` // Yen
	GBP  float32    `datastore:"gbp,noindex" json:"GBP"` // Pound Sterling
	USD  float32    `datastore:"usd,noindex" json:"USD"` // US Dollar
	EUR  float32    `datastore:"eur,noindex" json:"EUR"` // Euro
	CNY  float32    `datastore:"cny,noindex" json:"CNY"` // Yuan Renminbi
	CHF  float32    `datastore:"chf,noindex" json:"CHF"` // Swiss Franc
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