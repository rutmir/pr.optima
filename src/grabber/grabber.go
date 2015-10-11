package main

import (
	"log"
	"net/http"
	"encoding/json"
	"strconv"
	"time"
	"pr.optima/src/core/entities"
	"pr.optima/src/core/statistic"
	"pr.optima/src/repository"
	"pr.optima/src/server/rest.go/server/responses"
)
const (
	sourceUrl = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
	frameSize = 200
	step = 5
	rangeCount = 6
)
var (
	_rate responses.RateResponse
	_error responses.ErrorResponse
	_repo repository.RateRepo
)

func init() {
	_repo = repository.New(frameSize)

	_now := time.Now()
	_next := _now.Round(time.Hour)
	if _next.Hour() == _now.Hour() {
		_next = _next.Add(time.Hour)
	}
	_next = _next.Add(time.Second)
	//	ticker := time.NewTicker(_next.Sub(_now))
	ticker := time.NewTicker(time.Second)
	quit := make(chan struct {})

	for {
		select {
		case <-ticker.C:
			ticker.Stop()

			resp, err := http.Get(sourceUrl)
			if err != nil {
				log.Fatal(err)
				return
			}
			defer resp.Body.Close()

			now := time.Now()
			next := now.Round(time.Hour).Add(time.Hour).Add(time.Second)
			dec := json.NewDecoder(resp.Body)
			if resp.StatusCode == 200 {
				dec.Decode(&_rate)
				if err := _repo.Push(entities.Rate{
					Base    : _rate.Base,
					Id      : _rate.TimestampUnix,
					RUB     : _rate.Rates["RUB"],
					JPY     : _rate.Rates["JPY"],
					GBP     : _rate.Rates["GBP"],
					USD     : _rate.Rates["USD"],
					EUR     : _rate.Rates["EUR"],
					CNY     : _rate.Rates["CNY"],
					CHF     : _rate.Rates["CHF"]}); err != nil {
					log.Printf("Push rate to repo error: %v", err)
					next = now.Round(time.Minute).Add(time.Minute)
				}else {
					// logic here
					go executeDomainLogic()
				}
			} else {
				dec.Decode(&_error)
				log.Fatal(_error.ToString())
				return
			}
			ticker = time.NewTicker(next.Sub(now))
			log.Printf("Next tick: %v;\t Repo length: %v", next, strconv.Itoa(_repo.Len()))

		case <-quit:
			ticker.Stop()
			return
		}
	}
}

func executeDomainLogic() {
	l := _repo.Len()
	if l > frameSize + step {
		newL, err := _repo.Resize(frameSize)
		if err != nil {
			log.Printf("Repo resize error: %V", err)
		}else {
			//	re initialize nero nets
			log.Printf("Repo new length: %d", newL)
		}
	}
	r := extractFloatSet(_repo.GetAll(), "RUB")
	ranges, err := statistic.CalculateRanges(r, rangeCount)
	if err != nil {
		log.Print(err)
	}else {
		classes, err := statistic.CalculateClasses(r, ranges)
		if err != nil {
			log.Print(err)
		}else {
			log.Print(classes)
		}
	}
}

func extractFloatSet(rates []entities.Rate, symbol string) []float32 {
	var result []float32

	for _, element := range rates {
		switch symbol {
		case "RUB":
			result = append(result, element.RUB)
		}
	}

	return result
}

func main() {
}