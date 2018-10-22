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
	pushResultData commandResultDataAction = iota
	syncResultData
	getAllResultData
	getLastResultData
	getResultData
	lengthResultData
	resizeResultData
	reloadResultData
	clearResultData
	endResultData
)

type commandResultData struct {
	action    commandResultDataAction
	value     entities.ResultData
	size      int
	timestamp int64
	result    chan<- interface{}
	data      chan<- []entities.ResultData
	error     chan<- error
}

type singleResultData struct {
	value entities.ResultData
	found bool
}

type resultDataRepo struct {
	pipe       chan commandResultData
	symbol     string
	limit      int
	autoResize bool
	lastID     int64
	data       []entities.ResultData
	client     *datastore.Client
}

type commandResultDataAction int

// ResultDataRepo type
type ResultDataRepo interface {
	Push(entities.ResultData) error
	Sync(entities.ResultData) error
	Len() int
	GetAll() []entities.ResultData
	GetLast() (entities.ResultData, bool)
	Get(int64) (entities.ResultData, bool)
	Close() []entities.ResultData
	Resize(int) (int, error)
	Reload() (int, error)
	Clear(int64) error
}

// Push add new ResultData to repo
func (rr *resultDataRepo) Push(value entities.ResultData) error {
	reply := make(chan error)
	rr.pipe <- commandResultData{action: pushResultData, value: value, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	}
	return nil
}

// Sync repo
func (rr *resultDataRepo) Sync(value entities.ResultData) error {
	reply := make(chan error)
	rr.pipe <- commandResultData{action: syncResultData, value: value, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	}
	return nil
}

// Len length of the repo
func (rr *resultDataRepo) Len() int {
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: lengthResultData, result: reply}
	return (<-reply).(int)
}

// GetAll - return all stored data
func (rr *resultDataRepo) GetAll() []entities.ResultData {
	reply := make(chan []entities.ResultData)
	rr.pipe <- commandResultData{action: getAllResultData, data: reply}
	return <-reply
}

// Close repo
func (rr *resultDataRepo) Close() []entities.ResultData {
	reply := make(chan []entities.ResultData)
	rr.pipe <- commandResultData{action: endResultData, data: reply}
	return <-reply
}

// GetLast retrun the last item from repo
func (rr *resultDataRepo) GetLast() (entities.ResultData, bool) {
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: getLastResultData, result: reply}
	result := (<-reply).(singleResultData)
	return result.value, result.found
}

// Get return item by timestamp
func (rr *resultDataRepo) Get(timestamp int64) (entities.ResultData, bool) {
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: getResultData, timestamp: timestamp, result: reply}
	result := (<-reply).(singleResultData)
	return result.value, result.found
}

// Resize chanhe size of the repo
func (rr *resultDataRepo) Resize(size int) (int, error) {
	if size < 0 {
		return -1, fmt.Errorf("size parameter: %d must be positive value", size)
	}
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: resizeResultData, size: size, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	}
	return result, nil
}

// Reload update cached repo newest data
func (rr *resultDataRepo) Reload() (int, error) {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: reloadResultData, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	}
	return result, nil
}

// Clear remove data from repo
func (rr *resultDataRepo) Clear(date int64) error {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: clearResultData, error: errReply, result: reply, timestamp: date}
	err := <-errReply
	if err != nil {
		return error(err)
	}
	return nil
}

func (rr *resultDataRepo) run() {
	for command := range rr.pipe {
		switch command.action {
		case pushResultData:
			if command.value.Timestamp < rr.lastID+2500 {
				command.error <- fmt.Errorf("shift required (last: %d, new: %d)", rr.lastID, command.value.Timestamp)
				continue
			}
			_, err := rr.insertNewResultData(command.value)
			if err == nil {
				rr.lastID = command.value.Timestamp
				rr.data = append(rr.data, command.value)
				l := len(rr.data)
				if l > rr.limit && rr.autoResize {
					rr.data = rr.data[l-rr.limit:]
				}
			}
			command.error <- err
		case syncResultData:
			//	find and updated in local dataset
			key := command.value.GetCompositeKey()
			found := false
			for i, item := range rr.data {
				if item.GetCompositeKey() == key {
					rr.data[i] = command.value
					found = true
					break
				}
			}
			if found {
				_, err := rr.insertNewResultData(command.value)
				command.error <- err
			} else {
				command.error <- fmt.Errorf("ResultDataRepo Sync error: local data with key '%s' not found", key)
			}
		case getAllResultData:
			command.data <- rr.data
		case getLastResultData:
			l := len(rr.data)
			if l > 0 {
				command.result <- singleResultData{value: rr.data[l-1], found: true}
			} else {
				command.result <- singleResultData{value: entities.ResultData{}, found: false}
			}
		case getResultData:
			found := false
			for _, item := range rr.data {
				if item.Timestamp == command.timestamp {
					command.result <- singleResultData{value: item, found: true}
					found = true
					break
				}
			}
			if !found {
				command.result <- singleResultData{value: entities.ResultData{}, found: false}
			}
		case lengthResultData:
			command.result <- len(rr.data)
		case resizeResultData:
			l := len(rr.data)
			if l >= command.size {
				rr.data = rr.data[l-command.size:]
				command.error <- nil
				command.result <- len(rr.data)
			} else {
				command.error <- fmt.Errorf("repo size: %d less than new size: %d", l, command.size)
				command.result <- -1
			}
		case reloadResultData:
			if err := rr.loadStartResultData(); err != nil {
				command.error <- fmt.Errorf("repo reload error: %v", err)
				command.result <- -1
			} else {
				command.error <- nil
				command.result <- len(_rates)
			}
		case clearResultData:
			if err := rr.clearDataRepo(command.timestamp); err != nil {
				command.error <- fmt.Errorf("repo clear error: %v", err)
			} else {
				command.error <- nil
			}
		case endResultData:
			close(rr.pipe)
			command.data <- rr.data
		}
	}
}

// NewResultDataRepo - return new instance of the ResultDataRepo
func NewResultDataRepo(limit int, autoResize bool, symbol string, r *http.Request) ResultDataRepo {
	var ctx context.Context
	if r != nil {
		ctx = appengine.NewContext(r)
	} else {
		ctx = context.Background()
	}

	client, err := datastore.NewClient(ctx, projectID)
	if err != nil {
		log.Fatal(err)
	}

	rr := new(resultDataRepo)
	rr.pipe = make(chan commandResultData)
	rr.symbol = symbol
	rr.limit = limit
	rr.autoResize = autoResize == true
	rr.client = client

	if err := rr.loadStartResultData(); err != nil {
		log.Fatal(err)
	}

	go rr.run()
	return rr
}

// Cloud datastore logic
func (rr *resultDataRepo) loadStartResultData() error {
	var dst []entities.ResultData
	if _, err := rr.client.GetAll(context.Background(), datastore.NewQuery("ResultData").Filter("symbol=", rr.symbol).Order("-timestamp").Limit(rr.limit), &dst); err != nil {
		log.Printf("loadStartResultData error: %v\n", err)
		//return err
	}
	if dst != nil {
		l := len(dst)
		rr.data = make([]entities.ResultData, l)
		idx := 0
		for i := l - 1; i > -1; i-- {
			rr.data[idx] = dst[i]
			idx++
		}
		rr.lastID = rr.data[idx-1].Timestamp
	}
	return nil
}

func (rr *resultDataRepo) insertNewResultData(data entities.ResultData) (*datastore.Key, error) {
	ctx := context.Background()
	return rr.client.Put(ctx, datastore.NewKey(ctx, "ResultData", data.GetCompositeKey(), 0, nil), &data)
}

func (rr *resultDataRepo) clearDataRepo(unixdate int64) error {
	ctx := context.Background()
	var keys []*datastore.Key
	var err error
	if keys, err = rr.client.GetAll(ctx, datastore.NewQuery("ResultData").Filter("symbol=", rr.symbol).Filter("timestamp<", unixdate).KeysOnly(), nil); err != nil {
		log.Printf("clearDataRepo error: %v", err)
		//return err
	}
	if keys != nil {
		return rr.client.DeleteMulti(ctx, keys)
	}
	return err
}
