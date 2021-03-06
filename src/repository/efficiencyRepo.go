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
	syncEfficiency commandEfficiencyAction = iota
	getAllEfficiency
	getLastEfficiency
	lengthEfficiency
	reloadEfficiency
	clearEfficiency
	endEfficiency
)

type commandEfficiency struct {
	action    commandEfficiencyAction
	value     entities.Efficiency
	size      int
	timestamp int64
	result    chan<- interface{}
	data      chan<- []entities.Efficiency
	error     chan<- error
}

type singleEfficiency struct {
	value entities.Efficiency
	found bool
}

type efficiencyRepo struct {
	pipe        chan commandEfficiency
	symbol      string
	limit       int32
	frame       int32
	rangesCount int32
	trainType   string
	data        []entities.Efficiency
	client      *datastore.Client
}

type commandEfficiencyAction int

// EfficiencyRepo - type for presentation repo of Efficiency entity
type EfficiencyRepo interface {
	Sync(entities.Efficiency) error
	Len() int
	GetAll() []entities.Efficiency
	GetLast() (entities.Efficiency, bool)
	Close() []entities.Efficiency
	Reload() (int, error)
	Clear(int64) error
}

func (rr *efficiencyRepo) Sync(value entities.Efficiency) error {
	reply := make(chan error)
	rr.pipe <- commandEfficiency{action: syncEfficiency, value: value, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	}
	return nil
}

func (rr *efficiencyRepo) Len() int {
	reply := make(chan interface{})
	rr.pipe <- commandEfficiency{action: lengthEfficiency, result: reply}
	return (<-reply).(int)
}

func (rr *efficiencyRepo) GetAll() []entities.Efficiency {
	reply := make(chan []entities.Efficiency)
	rr.pipe <- commandEfficiency{action: getAllEfficiency, data: reply}
	return <-reply
}

func (rr *efficiencyRepo) Close() []entities.Efficiency {
	reply := make(chan []entities.Efficiency)
	rr.pipe <- commandEfficiency{action: endEfficiency, data: reply}
	return <-reply
}

func (rr *efficiencyRepo) GetLast() (entities.Efficiency, bool) {
	reply := make(chan interface{})
	rr.pipe <- commandEfficiency{action: getLastEfficiency, result: reply}
	result := (<-reply).(singleEfficiency)
	return result.value, result.found
}

func (rr *efficiencyRepo) Reload() (int, error) {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandEfficiency{action: reloadEfficiency, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	}
	return result, nil
}

func (rr *efficiencyRepo) Clear(date int64) error {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandEfficiency{action: clearEfficiency, error: errReply, result: reply, timestamp: date}
	err := <-errReply
	if err != nil {
		return error(err)
	}
	return nil
}

func (rr *efficiencyRepo) run() {
	for command := range rr.pipe {
		switch command.action {
		case syncEfficiency:
			_, err := rr.insertNewEfficiency(command.value)
			if err == nil {
				key := command.value.GetCompositeKey()
				found := false
				for i, item := range rr.data {
					if item.GetCompositeKey() == key {
						rr.data[i] = command.value
						found = true
						break
					}
				}
				if !found {
					rr.data = append(rr.data, command.value)
				}
			}
			command.error <- err
		case getAllEfficiency:
			command.data <- rr.data
		case getLastEfficiency:
			l := len(rr.data)
			if l > 0 {
				command.result <- singleEfficiency{value: rr.data[l-1], found: true}
			} else {
				command.result <- singleEfficiency{value: entities.Efficiency{TrainType: rr.trainType, Symbol: rr.symbol, RangesCount: rr.rangesCount, Limit: rr.limit, Frame: rr.frame}, found: false}
			}
		case lengthEfficiency:
			command.result <- len(rr.data)
		case reloadEfficiency:
			if err := rr.loadStartEfficiency(); err != nil {
				command.error <- fmt.Errorf("Repo reload error: %v.", err)
				command.result <- -1
			} else {
				command.error <- nil
				command.result <- len(_rates)
			}
		case clearEfficiency:
			if err := rr.clearEfficiency(command.timestamp); err != nil {
				command.error <- fmt.Errorf("Repo clear error: %v.", err)
			} else {
				command.error <- nil
			}
		case endEfficiency:
			close(rr.pipe)
			command.data <- rr.data
		}
	}
}

// NewEfficiencyRepo return instance of the EfficiencyRepo
func NewEfficiencyRepo(trainType, symbol string, rangesCount, limit, frame int32, r *http.Request) EfficiencyRepo {
	// todo: switch from http.Request to context.Context
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

	rr := new(efficiencyRepo)
	rr.pipe = make(chan commandEfficiency)
	rr.symbol = symbol
	rr.trainType = trainType
	rr.limit = limit
	rr.frame = frame
	rr.rangesCount = rangesCount
	rr.client = client

	if err := rr.loadStartEfficiency(); err != nil {
		log.Fatal(err)
	}

	go rr.run()
	return rr
}

// Cloud datastore logic
func (rr *efficiencyRepo) loadStartEfficiency() error {
	var dst []entities.Efficiency
	// todo: why context.Background() for appengine
	if _, err := rr.client.GetAll(context.Background(), datastore.NewQuery("Efficiency").Filter("symbol=", rr.symbol).Filter("trainType=", rr.trainType).Filter("rangesCount=", rr.rangesCount).Filter("limit=", rr.limit).Filter("frame=", rr.frame), &dst); err != nil {
		log.Printf("loadStartEfficiency error: %v", err)
		//		return err
	}
	if dst != nil {
		l := len(dst)
		rr.data = make([]entities.Efficiency, l)
		idx := 0
		for i := l - 1; i > -1; i-- {
			rr.data[idx] = dst[i]
			idx++
		}
	}
	return nil
}

func (rr *efficiencyRepo) insertNewEfficiency(data entities.Efficiency) (*datastore.Key, error) {
	ctx := context.Background()
	return rr.client.Put(ctx, datastore.NewKey(ctx, "Efficiency", data.GetCompositeKey(), 0, nil), &data)
}

func (rr *efficiencyRepo) clearEfficiency(unixdate int64) error {
	ctx := context.Background()
	var keys []*datastore.Key
	var err error
	if keys, err = rr.client.GetAll(ctx, datastore.NewQuery("Efficiency").Filter("symbol=", rr.symbol).Filter("trainType=", rr.trainType).Filter("rangesCount=", rr.rangesCount).Filter("limit=", rr.limit).Filter("frame=", rr.frame).Filter("timestamp<", unixdate).KeysOnly(), nil); err != nil {
		log.Printf("clearEfficiency error: %v", err)
		//return err
	}
	if keys != nil {
		return rr.client.DeleteMulti(ctx, keys)
	}
	return err
}
