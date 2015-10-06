package repository

import (
	"fmt"
	"io/ioutil"
	"log"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/datastore"

	"pr.optima/src/core/entities"
)

const (
	push commandAction = iota
	getAll
	getLast
	length
	end
)
const (
	projectID = "rp-optima"
	king = "Rate"
)
var (
	_step int = 5
	_highLimit int = _step * 11
	_lowLimit int = _step * 10
	_lastId int64
	_rates []entities.Rate
	_client *datastore.Client
)

type commandData struct {
	action commandAction
	value  entities.Rate
	result chan <- interface{}
	data   chan <- []entities.Rate
	error  chan <- error
}
type singleResult struct {
	value entities.Rate
	found bool
}
type rateRepo chan commandData
type commandAction int

type RateRepo interface {
	Push(entities.Rate) error
	Len() int
	GetAll() []entities.Rate
	GetLast() (entities.Rate, bool)
	Close() []entities.Rate
}

func (rr rateRepo) Push(rate entities.Rate) error {
	reply := make(chan error)
	rr <- commandData{action: push, value: rate, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	} else {
		return nil
	}
}
func (rr rateRepo) Len() int {
	reply := make(chan interface{})
	rr <- commandData{action: length, result: reply}
	return (<-reply).(int)
}
func (rr rateRepo) GetAll() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: getAll, data: reply}
	return <-reply
}
func (rr rateRepo) Close() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: end, data: reply}
	return <-reply
}
func (rr rateRepo) GetLast() (entities.Rate, bool) {
	reply := make(chan interface{})
	rr <- commandData{action: getLast, result: reply}
	result := (<-reply).(singleResult)
	return result.value, result.found
}
func (rr rateRepo) run() {
	for command := range rr {
		switch command.action {
		case push:
			if command.value.Id < _lastId + 3500 {
				command.error <- fmt.Errorf("Hour shift required (last: %d, new: %d).", _lastId, command.value.Id)
				continue
			}
			_, err := insertNewRate(command.value)
			if err == nil {
				_lastId = command.value.Id
				_rates = append(_rates, command.value)
				var l = len(_rates)
				if l > _highLimit {
					_rates = _rates[l - _lowLimit :]
					// ToDo: call rebuild statistics functionality here
				}
			}
			command.error <- err
		//	command.error <- saveToFile()
		case getAll:
			command.data <- _rates
		case getLast:
			l := len(_rates)
			if l > 0 {
				command.result <- singleResult{value:_rates[l - 1], found:true }
			} else {
				command.result <- singleResult{value: entities.Rate{}, found:false }
			}
		case length:
			command.result <- len(_rates)
		case end:
			close(rr)
			command.data <- _rates
		}
	}
}
func New() RateRepo {
	rr := make(rateRepo)
	go rr.run()
	return rr
}

type RateRepoPush struct {
	Rate   entities.Rate
	Result chan <- RateRepoPushResult
}
type RateRepoPushResult struct {
	Rate  entities.Rate
	Error error
}

func init() {
	jsonKey, err := ioutil.ReadFile("service-account.key.json")
	if err != nil {
		log.Fatal(err)
	}
	conf, err := google.JWTConfigFromJSON(
		jsonKey,
		datastore.ScopeDatastore,
		datastore.ScopeUserEmail,
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, projectID, cloud.WithTokenSource(conf.TokenSource(ctx)))
	if err != nil {
		log.Fatal(err)
	}
	_client = client

	if err := loadStartData(); err != nil {
		log.Fatal(err)
	}
}

func loadStartData() error {
	var dst []entities.Rate
	if _, err := _client.GetAll(context.Background(), datastore.NewQuery(king).Order("-id").Limit(_lowLimit), &dst); err != nil {
		return err
	}
	if dst != nil {
		for i := len(dst) - 1; i > -1; i-- {
			_rates = append(_rates, dst[i])
		}
		idx := len(_rates)
		_lastId = _rates[idx - 1].Id
	}
	return nil
}

func insertNewRate(rate entities.Rate) (*datastore.Key, error) {
	ctx := context.Background()
	return _client.Put(ctx, datastore.NewKey(ctx, king, "", rate.Id, nil), &rate)
}
