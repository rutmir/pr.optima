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
	pushRate commandAction = iota
	getAllRates
	getLastRate
	lengthRates
	resizeRates
	reloadRates
	endRates
)
const (
	projectID = "rp-optima"
	king = "Rate"
)
var (
	_limit int
	_autoResize bool
	_lastId int64
	_rates []entities.Rate
	_client *datastore.Client
)

type commandData struct {
	action commandAction
	value  entities.Rate
	size   int
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
	Resize(int) (int, error)
	Reload() (int, error)
}

func (rr rateRepo) Push(rate entities.Rate) error {
	reply := make(chan error)
	rr <- commandData{action: pushRate, value: rate, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	} else {
		return nil
	}
}
func (rr rateRepo) Len() int {
	reply := make(chan interface{})
	rr <- commandData{action: lengthRates, result: reply}
	return (<-reply).(int)
}
func (rr rateRepo) GetAll() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: getAllRates, data: reply}
	return <-reply
}
func (rr rateRepo) Close() []entities.Rate {
	reply := make(chan []entities.Rate)
	rr <- commandData{action: endRates, data: reply}
	return <-reply
}
func (rr rateRepo) GetLast() (entities.Rate, bool) {
	reply := make(chan interface{})
	rr <- commandData{action: getLastRate, result: reply}
	result := (<-reply).(singleResult)
	return result.value, result.found
}
func (rr rateRepo) Resize(size int) (int, error) {
	if size < 0 {
		return -1, fmt.Errorf("Size parameter: %d must be positive value.", size)
	}
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandData{action: resizeRates, size: size, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	} else {
		return result, nil
	}
}
func (rr rateRepo) Reload() (int, error) {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandData{action: reloadRates, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	} else {
		return result, nil
	}
}
func (rr rateRepo) run() {
	for command := range rr {
		switch command.action {
		case pushRate:
			if command.value.Id < _lastId + 2500 {
				command.error <- fmt.Errorf("Shift required (last: %d, new: %d).", _lastId, command.value.Id)
				continue
			}
			_, err := insertNewRate(command.value)
			if err == nil {
				_lastId = command.value.Id
				_rates = append(_rates, command.value)
				l := len(_rates)
				if l > _limit && _autoResize {
					_rates = _rates[l - _limit :]
				}
			}
			command.error <- err
		case getAllRates:
			command.data <- _rates
		case getLastRate:
			l := len(_rates)
			if l > 0 {
				command.result <- singleResult{value:_rates[l - 1], found:true }
			} else {
				command.result <- singleResult{value: entities.Rate{}, found:false }
			}
		case lengthRates:
			command.result <- len(_rates)
		case resizeRates:
			l := len(_rates)
			if l >= command.size {
				_rates = _rates[l - command.size :]
				command.error <- nil
				command.result <- len(_rates)
			}else {
				command.error <- fmt.Errorf("Repo size: %d less than new size: %d.", l, command.size)
				command.result <- -1
			}
		case reloadRates:
			if err := loadStartRates(); err != nil {
				command.error <- fmt.Errorf("Repo reload error: %v.", err)
				command.result <- -1
			}else {
				command.error <- nil
				command.result <- len(_rates)
			}
		case endRates:
			close(rr)
			command.data <- _rates
		}
	}
}
func New(limit int, autoResize bool) RateRepo {
	_limit = limit
	_autoResize = autoResize == true
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

	if err := loadStartRates(); err != nil {
		log.Fatal(err)
	}

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

// Cloud datastore logic
func loadStartRates() error {
	var dst []entities.Rate
	if _, err := _client.GetAll(context.Background(), datastore.NewQuery(king).Order("-id").Limit(_limit), &dst); err != nil {
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
