package main
import (
	"log"
	"time"
	"net/http"
	"encoding/json"

	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
	"pr.optima/src/grabber/work"
	"fmt"
)

const (
	source1Url = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
	source2Url = "http://www.apilayer.net/api/live?access_key=85c7d5e8f98fe83fa3fa81aafe489022&currencies=RUB,JPY,GBP,USD,EUR,CNY,CHF" //"85c7d5e8f98fe83fa3fa81aafe489022"
	repoSize = 200
)
var (
	_repo repository.RateRepo
	_rubWork *work.Work
	_eurWork *work.Work
	_gbpWork *work.Work
)

func init() {
	_repo = repository.New(repoSize, true)
	_rubWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "RUB")
	_eurWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "EUR")
	_gbpWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "GBP")

	_now := time.Now()
	_next := _now.Round(time.Hour)
	if _next.Hour() == _now.Hour() {
		_next = _next.Add(time.Hour)
	}
	_next = _next.Add(time.Second * 10)
	log.Printf("Start tick: %v.", _next)
	ticker := time.NewTicker(_next.Sub(_now))
	// ticker := time.NewTicker(time.Second)
	quit := make(chan struct {})

	for {
		select {
		case <-ticker.C:
			ticker.Stop()
			timestamp, success, err := updateFromSource2()
			if err != nil {
				log.Fatal(err)
				return
			}
			now := time.Now()
		//	next := now.Round(time.Hour).Add(time.Hour).Add(time.Second * 10)
			next := time.Unix(timestamp, 0).Add(time.Hour)

			if success == false {
				next = now.Round(time.Minute).Add(time.Minute * 5)
			}else {
				// logic here
				go executeDomainLogic()
			}

			ticker = time.NewTicker(next.Sub(now))
			log.Printf("Next tick: %v;\t Repo length: %d.", next, _repo.Len())

		case <-quit:
			ticker.Stop()
			return
		}
	}
}

func executeDomainLogic() {
	rates := _repo.GetAll()
	//	rub work
	if _rubWork.Limit < len(rates) {
		rub, err := _rubWork.Process(rates)

		if err != nil {
			log.Printf("Rub executeDomainLogic error: %v", err)
		}

		log.Printf("Rub nueral result: %d", rub)
	}

	if _eurWork.Limit < len(rates) {
		eur, err := _eurWork.Process(rates)

		if err != nil {
			log.Printf("Eur executeDomainLogic error: %v", err)
		}

		log.Printf("Eur nueral result: %d", eur)
	}

	if _gbpWork.Limit < len(rates) {
		gbp, err := _gbpWork.Process(rates)

		if err != nil {
			log.Printf("Gbp executeDomainLogic error: %v", gbp)
		}

		log.Printf("Gbp nueral result: %d", gbp)
	}
}

func updateFromSource1() (bool, error) {
	resp, err := http.Get(source1Url)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	dec := json.NewDecoder(resp.Body)
	if resp.StatusCode == 200 {
		var rate entities.RateResponse
		dec.Decode(&rate)
		log.Println(rate.ToShortString())
		if err := _repo.Push(entities.Rate{
			Base    : rate.Base,
			Id      : rate.TimestampUnix,
			RUB     : rate.Rates["RUB"],
			JPY     : rate.Rates["JPY"],
			GBP     : rate.Rates["GBP"],
			USD     : rate.Rates["USD"],
			EUR     : rate.Rates["EUR"],
			CNY     : rate.Rates["CNY"],
			CHF     : rate.Rates["CHF"]}); err != nil {
			log.Printf("Push rate to repo error: %v.", err)
			return false, nil
		}
	}else {
		var error entities.ErrorResponse
		dec.Decode(&error)
		return false, fmt.Errorf(error.ToString())
	}
	return true, nil
}

func updateFromSource2() (int64, bool, error) {
	var result int64 = 0
	resp, err := http.Get(source2Url)
	if err != nil {
		return 0, false, err
	}
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	if resp.StatusCode == 200 {
		var rate entities.Rate2Response
		dec.Decode(&rate)
		result = rate.TimestampUnix
		log.Println(rate.ToShortString())
		if err := _repo.Push(entities.Rate{
			Base    : rate.Base,
			Id      : rate.TimestampUnix,
			RUB     : rate.Quotes["USDRUB"],
			JPY     : rate.Quotes["USDJPY"],
			GBP     : rate.Quotes["USDGBP"],
			USD     : rate.Quotes["USDUSD"],
			EUR     : rate.Quotes["USDEUR"],
			CNY     : rate.Quotes["USDCNY"],
			CHF     : rate.Quotes["USDCHF"]}); err != nil {
			log.Printf("Push rate to repo error: %v.", err)
			return 0, false, nil
		}
	}else {
		var error entities.Error2Response
		dec.Decode(&error)
		return 0, false, fmt.Errorf(error.ToString())
	}
	return result, true, nil
}

func main() {}