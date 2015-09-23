package repository

import (
	"os"
	"fmt"
	"encoding/json"
	"../../entities"
)


var _dataStoreFileName string
var _limit int
var _lastId int64
var _rates []entities.Rate

type saveObject struct {
	Limit int             `json:"limit"`
	Rates []entities.Rate `json:"rates"`
}

type RateRepo interface {
	Push(entities.Rate) error
	Len() int
	GetAll() []entities.Rate
	GetLast() (entities.Rate, bool)
	Close() []entities.Rate
}
type rateRepo chan commandData
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
type commandAction int
const (
	push commandAction = iota
	getAll
	getLast
	length
	end
)

func (rr rateRepo) Push(rate entities.Rate) error {
	reply := make(chan error)
	rr <- commandData{action: push, value: rate, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	} else {
		return nil
	}
	//	return (<-reply).(error)
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
			_lastId = command.value.Id
			_rates = append(_rates, command.value)
			var l = len(_rates)
			if l > _limit {
				_rates = _rates[l - _limit:]
			}
			command.error <- saveToFile()
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


/*var _in     chan RateRepoPush*/

func init() {
	_dataStoreFileName = "ratesData.json"
	_limit = 9
	/*_in = make(chan RateRepoPush)*/
	/*RepoCreateTodo(entities.T_odo{Name: "Write presentation"})
	RepoCreateTodo(entities.T_odo{Name: "Host meetup"})*/
	/*go func() {
		for ratepush := range _in {
			rate, err := _PushNewRate(ratepush.Rate)
			ratepush.Result <- RateRepoPushResult{Rate : rate, Error : err }
		}
	}()*/
}

func saveToFile() error {
	dec, err := json.Marshal(saveObject{Limit: _limit, Rates:_rates})

	if err != nil {
		return err
	}
	if file, err := os.Create(_dataStoreFileName); err == nil {
		defer file.Close()
		if _, err := file.Write(dec); err != nil {
			return err
		}
	} else {
		return err
	}
	return nil
}

/*
func RepoFindTodo(id int) entities.T_odo {
	for _, t := range Todos {
		if t.Id == id {
			return t
		}
	}
	// return empty Todo_ if not found
	return entities.T_odo{}
}*/

/*go func() {
	for {
		select {
		case ratepush := <-_in
		rate, err := _PushNewRate(ratepush.Rate)
		ratepush.Result <- RateRepoPushResult{ Rate = rate, Error = err }
	}
}
}()*/
/*
func PushNewRate(r entities.Rate) <-chan RateRepoPushResult {
	result := make(chan RateRepoPushResult)
	_in <- RateRepoPush{Rate : r, Result: result}
	return result
}

func _PushNewRate(r entities.Rate) (entities.Rate, error) {
	if r.Id < _lastId + 3500 {
		return r, fmt.Errorf("Hour shift required (last: %d, new: %d).", _lastId, r.Id)
	}
	_lastId = r.Id
	_rates = append(_rates, r)
	var l = len(_rates)
	if l > _limit {
		_rates = _rates[l - _limit:]
	}
	fmt.Printf("len(_rates): %d", len(_rates))
	fmt.Println("")
	return r, nil
}*/

/*
func RepoDestroyTodo(id int) error {
	for i, t := range Todos {
		if t.Id == id {
			Todos = append(Todos[:i], Todos[i + 1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find T_odo with id of %d to delete", id)
}*/
