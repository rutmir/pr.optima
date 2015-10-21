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
	pushResultData commandResultDataAction = iota
	getAllResultData
	getLastResultData
	lengthResultData
	resizeResultData
	reloadResultData
	endResultData
)
type commandResultData struct {
	action commandResultDataAction
	value  entities.ResultData
	size   int
	result chan <- interface{}
	data   chan <- []entities.ResultData
	error  chan <- error
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
	lastId     int64
	data       []entities.ResultData
	client     *datastore.Client
}
type commandResultDataAction int

type ResultDataRepo interface {
	Push(entities.ResultData) error
	Len() int
	GetAll() []entities.ResultData
	GetLast() (entities.ResultData, bool)
	Close() []entities.ResultData
	Resize(int) (int, error)
	Reload() (int, error)
}

func (rr resultDataRepo) Push(value entities.ResultData) error {
	reply := make(chan error)
	rr.pipe <- commandResultData{action: pushResultData, value: value, error: reply}
	err := <-reply
	if err != nil {
		return error(err)
	} else {
		return nil
	}
}
func (rr resultDataRepo) Len() int {
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: lengthResultData, result: reply}
	return (<-reply).(int)
}
func (rr resultDataRepo) GetAll() []entities.ResultData {
	reply := make(chan []entities.ResultData)
	rr.pipe <- commandResultData{action: getAllResultData, data: reply}
	return <-reply
}
func (rr resultDataRepo) Close() []entities.ResultData {
	reply := make(chan []entities.ResultData)
	rr.pipe <- commandResultData{action: endResultData, data: reply}
	return <-reply
}
func (rr resultDataRepo) GetLast() (entities.ResultData, bool) {
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: getLastResultData, result: reply}
	result := (<-reply).(singleResultData)
	return result.value, result.found
}
func (rr resultDataRepo) Resize(size int) (int, error) {
	if size < 0 {
		return -1, fmt.Errorf("Size parameter: %d must be positive value.", size)
	}
	errReply := make(chan error)
	reply := make(chan interface{})
	rr.pipe <- commandResultData{action: resizeResultData, size: size, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	} else {
		return result, nil
	}
}
func (rr resultDataRepo) Reload() (int, error) {
	errReply := make(chan error)
	reply := make(chan interface{})
	rr <- commandResultData{action: reloadResultData, error: errReply, result: reply}
	err := <-errReply
	result := (<-reply).(int)
	if err != nil {
		return -1, error(err)
	} else {
		return result, nil
	}
}
func (rr resultDataRepo) run() {
	for command := range rr.pipe {
		switch command.action {
		case pushResultData:
			if command.value.Timestamp < rr.lastId + 3500 {
				command.error <- fmt.Errorf("Hour shift required (last: %d, new: %d).", rr.lastId, command.value.Timestamp)
				continue
			}
			_, err := rr.insertNewResultData(command.value)
			if err == nil {
				rr.lastId = command.value.Timestamp
				rr.data = append(rr.data, command.value)
				l := len(rr.data)
				if l > rr.limit && rr.autoResize {
					rr.data = rr.data[l - rr.limit :]
				}
			}
			command.error <- err
		case getAllResultData:
			command.data <- rr.data
		case getLastResultData:
			l := len(rr.data)
			if l > 0 {
				command.result <- singleResultData{value:rr.data[l - 1], found:true }
			} else {
				command.result <- singleResultData{value:entities.ResultData{}, found:false }
			}
		case lengthResultData:
			command.result <- len(rr.data)
		case resizeResultData:
			l := len(rr.data)
			if l >= command.size {
				rr.data = rr.data[l - command.size :]
				command.error <- nil
				command.result <- len(rr.data)
			}else {
				command.error <- fmt.Errorf("Repo size: %d less than new size: %d.", l, command.size)
				command.result <- -1
			}
		case reloadResultData:
			if err := rr.loadStartResultData(); err != nil {
				command.error <- fmt.Errorf("Repo reload error: %v.", err)
				command.result <- -1
			}else {
				command.error <- nil
				command.result <- len(_rates)
			}
		case endResultData:
			close(rr.pipe)
			command.data <- rr.data
		}
	}
}
func NewResultDataRepo(limit int, autoResize bool, symbol string) ResultDataRepo {
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
func (rr resultDataRepo) loadStartResultData() error {
	var dst []entities.ResultData
	if _, err := rr.client.GetAll(context.Background(), datastore.NewQuery("ResultData").Filter("symbol=", rr.symbol).Order("-timestamp").Limit(rr.limit), &dst); err != nil {
		return err
	}
	if dst != nil {
		for i := len(dst) - 1; i > -1; i-- {
			rr.data = append(rr.data, dst[i])
		}
		idx := len(rr.data)
		rr.lastId = rr.data[idx - 1].Timestamp
	}
	return nil
}
func (rr resultDataRepo) insertNewResultData(data entities.ResultData) (*datastore.Key, error) {
	ctx := context.Background()
	return rr.client.Put(ctx, datastore.NewKey(ctx, "ResultData", data.GetCompositeKey(), 0, nil), &data)
}
