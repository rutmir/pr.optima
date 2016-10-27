package entities

import (
	"fmt"
	"strings"
	"time"
)

// Rate struct
type Rate struct {
	Base string  `datastore:"base,noindex" json:"base"`
	ID   int64   `datastore:"id,index" json:"timestamp"`
	RUB  float32 `datastore:"rub,noindex" json:"RUB"` // Russian Ruble
	JPY  float32 `datastore:"jpy,noindex" json:"JPY"` // Yen
	GBP  float32 `datastore:"gbp,noindex" json:"GBP"` // Pound Sterling
	USD  float32 `datastore:"usd,noindex" json:"USD"` // US Dollar
	EUR  float32 `datastore:"eur,noindex" json:"EUR"` // Euro
	CNY  float32 `datastore:"cny,noindex" json:"CNY"` // Yuan Renminbi
	CHF  float32 `datastore:"chf,noindex" json:"CHF"` // Swiss Franc
}

// ToString method
func (f *Rate) ToString() string {
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

// Timestamp method
func (f *Rate) Timestamp() time.Time {
	return time.Unix(f.ID, 0).UTC()
}

// SetTimestamp method
func (f *Rate) SetTimestamp(timestamp time.Time) {
	f.ID = timestamp.Unix()
}

// GetForSymbol method
func (f *Rate) GetForSymbol(symbol string) (float32, error) {
	if len(symbol) != 3 {
		return -1, fmt.Errorf("Unknowen symbol: '%v'", symbol)
	}
	switch strings.ToUpper(symbol) {
	case "RUB":
		return f.RUB, nil
	case "JPY":
		return f.JPY, nil
	case "GBP":
		return f.GBP, nil
	case "USD":
		return f.USD, nil
	case "EUR":
		return f.EUR, nil
	case "CNY":
		return f.CNY, nil
	case "CHF":
		return f.CHF, nil
	}
	return -1, fmt.Errorf("Unknowen symbol: '%v'", symbol)
}
