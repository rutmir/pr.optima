package repository

import (
	"fmt"
	"log"
	"net/http"

	"cloud.google.com/go/datastore"
	"golang.org/x/net/context"
	"google.golang.org/appengine"

	"pr.optima/src/core/entities"
)

const (
	pushRate commandAction = iota
	getAllRates
	getLastRate
	lengthRates
	resizeRates
	reloadRates
	clearRates
	endRates
)

const (
	projectID = "rp-optima"
	king      = "Rate"
)

var (
	_limit      int
	_autoResize bool
	_lastID     int64
	_rates      []entities.Rate
	_client     *datastore.Client
	//_ctx        context.Context
)

type commandData struct {
	action    commandAction
	value     entities.Rate
	size      int
	timestamp int64
	result    chan<- interface{}
	data      chan<- []entities.Rate
	error     chan<- error
}

type singleResult struct {
	value entities.Rate
	found bool
}

type rateRepo chan commandData

type commandAction int

// RateRepo - type of presentation repo for Rate
type RateRepo interface {
	Push(entities.Rate) error
	Len() int
	GetAll() []entities.Rate
	GetLast() (entities.Rate, bool)
	Close() []entities.Rate
	Resize(int) (int, error)
	Reload() (int, error)
	Clear(int64) error
}

// Push - add rate to repo
func (rr rateRepo) Push(rate entities.Rate) error {
	reply := make(chan error)
	rr <- commandData{action: pushRate, value: rate, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	}
	return nil
}

// Len return length of stored data
func (rr rateRepo) Len() int {
	reply := make(chan interface{})
	rr <- commandData{action: lengthRates, result: reply}
	return (<-reply).(int)
}

// GetAll return array of rates
func (rr rateRepo) GetAll() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: getAllRates, data: reply}
	return <-reply
}

// Close - close repo method
func (rr rateRepo) Close() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: endRates, data: reply}
	return <-reply
}

// GetLast - return the last Rate from repo
func (rr rateRepo) GetLast() (entities.Rate, bool) {
	reply := make(chan interface{})
	rr <- commandData{action: getLastRate, result: reply}
	result := (<-reply).(singleResult)
	return result.value, result.found
}

// Resize - resize repo length
func (rr rateRepo) Resize(size int) (int, error) {
	if size < 0 {
		return -1, fmt.Errorf("size parameter: %d must be positive value", size)
	}
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandData{action: resizeRates, size: size, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	}
	return result, nil
}

// Reload - update in cache data from newest data
func (rr rateRepo) Reload() (int, error) {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandData{action: reloadRates, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	}
	return result, nil
}

// Clear - clear in cache repo
func (rr rateRepo) Clear(date int64) error {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandData{action: clearRates, error: errReply, result: reply, timestamp: date}
	err := <-errReply
	if err != nil {
		return error(err)
	}
	return nil
}

func (rr rateRepo) run() {
	for command := range rr {
		switch command.action {
		case pushRate:
			if command.value.ID < (_lastID + 2500) {
				command.error <- fmt.Errorf("shift required (last: %d, new: %d)", _lastID, command.value.ID)
				continue
			}
			_, err := insertNewRate(command.value)
			if err == nil {
				_lastID = command.value.ID
				_rates = append(_rates, command.value)
				l := len(_rates)
				if l > _limit && _autoResize {
					_rates = _rates[l-_limit:]
				}
			}
			command.error <- err
		case getAllRates:
			command.data <- _rates
		case getLastRate:
			l := len(_rates)
			if l > 0 {
				command.result <- singleResult{value: _rates[l-1], found: true}
			} else {
				command.result <- singleResult{value: entities.Rate{}, found: false}
			}
		case lengthRates:
			command.result <- len(_rates)
		case resizeRates:
			l := len(_rates)
			if l >= command.size {
				_rates = _rates[l-command.size:]
				command.error <- nil
				command.result <- len(_rates)
			} else {
				command.error <- fmt.Errorf("repo size: %d less than new size: %d", l, command.size)
				command.result <- -1
			}
		case reloadRates:
			if err := loadStartRates(); err != nil {
				command.error <- fmt.Errorf("repo reload error: %v", err)
				command.result <- -1
			} else {
				command.error <- nil
				command.result <- len(_rates)
			}
		case clearRates:
			if err := fnClearRates(command.timestamp); err != nil {
				command.error <- fmt.Errorf("repo clear error: %v", err)
			} else {
				command.error <- nil
			}
		case endRates:
			close(rr)
			command.data <- _rates
		}
	}
}

// New - return new instance of repo
func New(limit int, autoResize bool, r *http.Request) RateRepo {
	_limit = limit
	_autoResize = autoResize == true

	var ctx context.Context
	if r != nil {
		ctx = appengine.NewContext(r)
		//_ctx = ctx
	} else {
		ctx = context.Background()
	}

	client, err := datastore.NewClient(ctx, projectID)
	if err != nil {
		log.Fatal(err)
	}
	_client = client

	if err := loadStartRates(); err != nil {
		log.Fatal(err)
	}

	rr := make(rateRepo)
	go rr.run()
	return rr
}

// RateRepoPush type
type RateRepoPush struct {
	Rate   entities.Rate
	Result chan<- RateRepoPushResult
}

// RateRepoPushResult type
type RateRepoPushResult struct {
	Rate  entities.Rate
	Error error
}

// Cloud datastore logic
func loadStartRates() error {
	var dst []entities.Rate
	if _, err := _client.GetAll(context.Background(), datastore.NewQuery(king).Order("-id").Limit(_limit), &dst); err != nil {
		log.Printf("loadStartRates error: %v\n", err)
		//return err
	}
	if dst != nil {
		l := len(dst)
		_rates = make([]entities.Rate, l)
		idx := 0
		for i := l - 1; i > -1; i-- {
			_rates[idx] = dst[i]
			idx++
		}
		_lastID = _rates[idx-1].ID
	}
	return nil
}

func insertNewRate(rate entities.Rate) (*datastore.Key, error) {
	ctx := context.Background()
	return _client.Put(ctx, datastore.NewKey(ctx, king, "", rate.ID, nil), &rate)
}

func fnClearRates(unixdate int64) error {
	ctx := context.Background()
	var keys []*datastore.Key
	var err error

	if keys, err = _client.GetAll(ctx, datastore.NewQuery(king).Filter("id<", unixdate).KeysOnly(), nil); err != nil {
		log.Printf("clearRates error: %v", err)
		return err
	}
	if keys != nil {
		return _client.DeleteMulti(ctx, keys)
	}
	return err
}
