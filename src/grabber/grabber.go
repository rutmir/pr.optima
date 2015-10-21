package main
import (
	"log"
	"time"
	"net/http"
	"encoding/json"

	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
	"./work"
)

const (
	sourceUrl = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
	repoSize = 200
)
var (
	_rate entities.RateResponse
	_error entities.ErrorResponse
	_repo repository.RateRepo
	_rubWork *work.Work
)

func init() {
	_repo = repository.New(repoSize, true)
	_rubWork = work.NewWork(6, 5, 20, 1, work.TTLbfgs, "RUB")

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
					log.Printf("Push rate to repo error: %v.", err)
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
			log.Printf("Next tick: %v;\t Repo length: %d.", next, _repo.Len())

		case <-quit:
			ticker.Stop()
			return
		}
	}
}

func executeDomainLogic() {
	reates := _repo.GetAll()
	rub, err := _rubWork.Process(reates)

	if err != nil {
		log.Printf("executeDomainLogic error: %v", err)
	}

	log.Printf("nueral result: %d", rub)
}

func main() {}