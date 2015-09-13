package entities
import "fmt"

type Cources struct {
	Id         int        `json:"id"`
	Base       string        `json:"base"`
	Timestamp  uint64        `json:"timestamp"`
	Disclaimer string        `json"disclaimer"`
	Rates      map[string]float32    `json"rates"`
}

func (f Cources) ToString() string {
	return fmt.Sprintf("Base; %s, Timestamp: %i", f.Base, f.Timestamp)
}