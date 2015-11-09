package main
import (
	"fmt"
	"log"
	"time"
	"net/http"
	"encoding/json"

	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
	"pr.optima/src/grabber/work"
)

const (
	source1Url = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
	source2Url = "http://www.apilayer.net/api/live?access_key=85c7d5e8f98fe83fa3fa81aafe489022&currencies=RUB,JPY,GBP,USD,EUR,CNY,CHF" //"85c7d5e8f98fe83fa3fa81aafe489022"
	appEngineUrl = "https://rp-optima.appspot.com/api/refresh"
	repoSize = 200
)
const _authKey = "B7C05147C5A34376B30CEF2F289FBB6C"
var (
	_repo repository.RateRepo
	_rubWork *work.Work
	_eurWork *work.Work
	_gbpWork *work.Work
	_jpyWork *work.Work
	_cnyWork *work.Work
	_chfWork *work.Work
)

func init() {
	_repo = repository.New(repoSize, true, nil)
	_rubWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "RUB")
	_eurWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "EUR")
	_gbpWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "GBP")
	_jpyWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "JPY")
	_cnyWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "CNY")
	_chfWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "CHF")

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
	if _rubWork.Limit < len(rates) {
		rub, err := _rubWork.Process(rates)

		if err != nil {
			log.Printf("RUB executeDomainLogic error: %v", err)
		}

		log.Printf("RUB nueral result: %d", rub)
	}

	if _eurWork.Limit < len(rates) {
		eur, err := _eurWork.Process(rates)

		if err != nil {
			log.Printf("EUR executeDomainLogic error: %v", err)
		}

		log.Printf("EUR nueral result: %d", eur)
	}

	if _gbpWork.Limit < len(rates) {
		gbp, err := _gbpWork.Process(rates)

		if err != nil {
			log.Printf("GBP executeDomainLogic error: %v", err)
		}

		log.Printf("GBP nueral result: %d", gbp)
	}

	if _jpyWork.Limit < len(rates) {
		jpy, err := _jpyWork.Process(rates)

		if err != nil {
			log.Printf("JPY executeDomainLogic error: %v", err)
		}

		log.Printf("JPY nueral result: %d", jpy)
	}

	if _cnyWork.Limit < len(rates) {
		cny, err := _cnyWork.Process(rates)

		if err != nil {
			log.Printf("CNY executeDomainLogic error: %v", err)
		}

		log.Printf("CNY nueral result: %d", cny)
	}

	if _chfWork.Limit < len(rates) {
		chf, err := _chfWork.Process(rates)

		if err != nil {
			log.Printf("CHF executeDomainLogic error: %v", err)
		}

		log.Printf("CHF nueral result: %d", chf)
	}

	// refresh appengine
	if req, err := http.NewRequest("GET", appEngineUrl, nil); err == nil {
		req.Header.Add("Auth", _authKey)
		if resp, err := http.DefaultClient.Do(req); err != nil {
			log.Printf("Refresh appengine Do Request error: %v", err)
		}else {
			defer resp.Body.Close()
		}
	}else {
		log.Printf("Refresh appengine NewRequest error: %v", err)
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