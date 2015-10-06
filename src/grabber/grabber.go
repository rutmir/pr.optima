package main

import (
	"log"
	"net/http"
	"encoding/json"
	"strconv"
	"time"
	"pr.optima/src/core/entities"
	"pr.optima/src/server/rest.go/server/responses"
	"pr.optima/src/repository"
)
const (
	sourceUrl = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
)

func init() {
	var _rate responses.RateResponse
	var _error responses.ErrorResponse
	var _repo = repository.New()

	_now := time.Now()
	_next := _now.Round(time.Hour)
	if _next.Hour() == _now.Hour() {
		_next = _next.Add(time.Hour)
	}
	ticker := time.NewTicker(_next.Sub(_now))
	quit := make(chan struct {})

	//	go func() {
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
			next := now.Round(time.Hour).Add(time.Hour)
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
	//	}()
}

func main() {
}
