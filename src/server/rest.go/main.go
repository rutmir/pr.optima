package main

import (
	"fmt"
	"log"
	"net/http"
	"encoding/json"
	"strconv"
	"time"
//	"io/ioutil"
	"rest.go/entities"
	"rest.go/server"
	"rest.go/server/responses"
	"rest.go/server/repository"
)

func main() {
	var _rate responses.RateResponse
	var _error responses.ErrorResponse
	var _repo = repository.New()

	resp, err := http.Get("https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD")
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	dec := json.NewDecoder(resp.Body)
	fmt.Println("repo length: " + strconv.Itoa(_repo.Len()))

	if last, found := _repo.GetLast(); found == true {
		fmt.Println(last.ToString())
	}

	if resp.StatusCode == 200 {
		dec.Decode(&_rate)
		/*fmt.Println(_rate.ToString())*/
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
			log.Fatal(err)
		}
	} else {
		dec.Decode(&_error)
		fmt.Print(_error.ToString())
	}
	fmt.Println("repo length: " + strconv.Itoa(_repo.Len()))
	if last, found := _repo.GetLast(); found == true {
		fmt.Println(last.ToString())
	}

	var i time.Duration = 1;
	ticker := time.NewTicker(time.Second * i)
	quit := make(chan struct {})

	go func() {
		for {
			select {
			case <-ticker.C:
				i = i + 1
				ticker.Stop()
				ticker = time.NewTicker(time.Second * i)
				fmt.Printf("next tick: %v", time.Now())
				fmt.Println("------")
			case <-quit:
				ticker.Stop()
				return
			}
		}
	}()

	fmt.Println("contunue")
	router := server.NewRouter()
	log.Fatal(http.ListenAndServe(":8080", router))
}
