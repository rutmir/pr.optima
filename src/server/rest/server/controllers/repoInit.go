package controllers

import (
	"fmt"
	//	"sync"
	"io/ioutil"
	"log"
	"net/http"

	"cloud.google.com/go/datastore"
	"golang.org/x/net/context"
	"google.golang.org/appengine"
	logAE "google.golang.org/appengine/log"

	"errors"

	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
)

const historyLimit = 100

var (
	_initialized = false
	_rateRepo    repository.RateRepo
	_rates       []entities.Rate
	// RUB
	_rubResultRepo repository.ResultDataRepo
	_rubEffRepo    repository.EfficiencyRepo
	_rubResultList *entities.ResultDataListResponse
	_rubResult     *entities.ResultDataResponse
	_rubSignal     *entities.Signal
	// EUR
	_eurResultRepo repository.ResultDataRepo
	_eurEffRepo    repository.EfficiencyRepo
	_eurResultList *entities.ResultDataListResponse
	_eurResult     *entities.ResultDataResponse
	_eurSignal     *entities.Signal
	// GBP
	_gbpResultRepo repository.ResultDataRepo
	_gbpEffRepo    repository.EfficiencyRepo
	_gbpResultList *entities.ResultDataListResponse
	_gbpResult     *entities.ResultDataResponse
	_gbpSignal     *entities.Signal
	// CHF
	_chfResultRepo repository.ResultDataRepo
	_chfEffRepo    repository.EfficiencyRepo
	_chfResultList *entities.ResultDataListResponse
	_chfResult     *entities.ResultDataResponse
	_chfSignal     *entities.Signal
	// CNY
	_cnyResultRepo repository.ResultDataRepo
	_cnyEffRepo    repository.EfficiencyRepo
	_cnyResultList *entities.ResultDataListResponse
	_cnyResult     *entities.ResultDataResponse
	_cnySignal     *entities.Signal
	// JPY
	_jpyResultRepo repository.ResultDataRepo
	_jpyEffRepo    repository.EfficiencyRepo
	_jpyResultList *entities.ResultDataListResponse
	_jpyResult     *entities.ResultDataResponse
	_jpySignal     *entities.Signal
)

func initializeRepo(r *http.Request) {
	_rateRepo = repository.New(historyLimit+5, false, r)
	_rates = _rateRepo.GetAll()

	_rubResultRepo = repository.NewResultDataRepo(historyLimit, false, "RUB", r)
	_rubEffRepo = repository.NewEfficiencyRepo("L-BFGS", "RUB", 6, 20, 5, r)

	_eurResultRepo = repository.NewResultDataRepo(historyLimit, false, "EUR", r)
	_eurEffRepo = repository.NewEfficiencyRepo("L-BFGS", "EUR", 6, 20, 5, r)

	_gbpResultRepo = repository.NewResultDataRepo(historyLimit, false, "GBP", r)
	_gbpEffRepo = repository.NewEfficiencyRepo("L-BFGS", "GBP", 6, 20, 5, r)

	_chfResultRepo = repository.NewResultDataRepo(historyLimit, false, "CHF", r)
	_chfEffRepo = repository.NewEfficiencyRepo("L-BFGS", "CHF", 6, 20, 5, r)

	_cnyResultRepo = repository.NewResultDataRepo(historyLimit, false, "CNY", r)
	_cnyEffRepo = repository.NewEfficiencyRepo("L-BFGS", "CNY", 6, 20, 5, r)

	_jpyResultRepo = repository.NewResultDataRepo(historyLimit, false, "JPY", r)
	_jpyEffRepo = repository.NewEfficiencyRepo("L-BFGS", "JPY", 6, 20, 5, r)

	_initialized = true
}

func rebuildData() error {
	var err error
	_rubResultList, _rubResult, _rubSignal, err = populateSet(_rubResultRepo, _rubEffRepo)

	if err != nil {
		return err
	}

	_eurResultList, _eurResult, _eurSignal, err = populateSet(_eurResultRepo, _eurEffRepo)

	if err != nil {
		return err
	}

	_gbpResultList, _gbpResult, _gbpSignal, err = populateSet(_gbpResultRepo, _gbpEffRepo)

	if err != nil {
		return err
	}

	_chfResultList, _chfResult, _chfSignal, err = populateSet(_chfResultRepo, _chfEffRepo)

	if err != nil {
		return err
	}

	_cnyResultList, _cnyResult, _cnySignal, err = populateSet(_cnyResultRepo, _cnyEffRepo)

	if err != nil {
		return err
	}

	_jpyResultList, _jpyResult, _jpySignal, err = populateSet(_jpyResultRepo, _jpyEffRepo)

	if err != nil {
		return err
	}

	return nil
}

func populateSet(resultRepo repository.ResultDataRepo, effRepo repository.EfficiencyRepo) (*entities.ResultDataListResponse, *entities.ResultDataResponse, *entities.Signal, error) {
	eff, found := effRepo.GetLast()
	if !found {
		return nil, nil, nil, errors.New("populateSet error, in get last EFF not found")
	}
	score10, score100 := get10_100Score(eff)
	results := resultRepo.GetAll()
	l := len(results)
	var resultSet []entities.ResultResponse
	limit := historyLimit
	if l < historyLimit {
		limit = l
	}
	resultSet = make([]entities.ResultResponse, limit)

	counter := 1
	for i := l - 1; i >= l-limit && i >= 0; i-- {
		result := new(entities.ResultResponse)
		result.Prediction = results[i].Prediction
		result.Result = results[i].Result
		result.Source = make([]int32, len(results[i].Source))
		copy(result.Source, results[i].Source)
		result.Timestamp = results[i].Timestamp
		result.Levels = results[i].RangesCount

		if err := populateDataFor(result.Timestamp, results[i].Step, results[i].Symbol, results, result); err != nil {
			return nil, nil, nil, fmt.Errorf("PopulateSet error, in populate data: %v.", err)
		}
		result.Symbol = results[i].Symbol
		resultSet[limit-counter] = *result
		counter++
	}
	var signal *entities.Signal
	if last, found := resultRepo.GetLast(); found {
		signal = new(entities.Signal)
		signal.RangesCount = last.RangesCount
		signal.Symbol = last.Symbol
		signal.Timestamp = last.Timestamp
		signal.Prediction = last.Prediction
		signal.Score10 = score10
		signal.Score100 = score100
	}
	retList := new(entities.ResultDataListResponse)
	retList.Data = resultSet
	retList.Score10 = score10
	retList.Score100 = score100

	retCurrent := new(entities.ResultDataResponse)
	retCurrent.Data = *(resultSet[len(resultSet)-1].Clone())
	retCurrent.Score10 = score10
	retCurrent.Score100 = score100

	return retList, retCurrent, signal, nil
}

func get10_100Score(eff entities.Efficiency) (float32, float32) {
	l := len(eff.LastSD)
	index := 100
	if l < index {
		index = l
	}

	var tenSum int32
	var hundredSum int32

	for i := 1; i <= index; i++ {
		item := eff.LastSD[l-i]
		if i <= 10 {
			tenSum += item
		}
		hundredSum += item
	}
	if index < 10 {
		return float32(tenSum) / float32(index), float32(hundredSum) / float32(index)
	}
	return float32(tenSum) / 10, float32(hundredSum) / float32(index)
}

func populateDataFor(timestamp int64, frame int32, symbol string, results []entities.ResultData, result *entities.ResultResponse) error {
	l := len(_rates)
	addExtendedData := false
	if _rates[l-1].ID > timestamp {
		addExtendedData = true
		frame++
	}
	result.Data = make([]float32, frame)
	result.Time = make([]int64, frame)

	var counter int32 = 1
	var prevRate entities.Rate
	for i := l - 1; i > 0 && counter <= frame; i-- {
		if _rates[i].ID <= timestamp {
			if addExtendedData {
				value, err := prevRate.GetForSymbol(symbol)
				if err != nil {
					return fmt.Errorf("ExtrudeDataFor error: %v", err)
				}
				result.Data[frame-counter] = value
				result.Time[frame-counter] = prevRate.ID
				counter++
				addExtendedData = false
			}
			value, err := _rates[i].GetForSymbol(symbol)
			if err != nil {
				return fmt.Errorf("ExtrudeDataFor error: %v", err)
			}
			result.Data[frame-counter] = value
			result.Time[frame-counter] = _rates[i].ID
			counter++
		}
		prevRate = _rates[i]
	}
	return nil
}

func clearDbData(unixtime int64, r *http.Request) []error {
	errors := make([]error, 0, 2)
	jsonKey, err := ioutil.ReadFile("service-account.key.json")
	if err != nil {
		_logEror(r, err)
		errors = append(errors, err)
		return errors
	}
	//wg := sync.WaitGroup{}
	//wg.Add(1)
	//wg.Done()
	//wg.Wait()

	resultDataChan := _getResultDataKeys(r, jsonKey, unixtime)
	var resultDataErrorChan <-chan error
	rateChan := _getRatesKeys(r, jsonKey, unixtime)
	var rateErrorChan <-chan error

	resultDataChanClose, rateChanClose := false, true
	errorDataChanClose, errorChanClose := false, true

	for {
		if resultDataChanClose && rateChanClose {
			break
		}
		select {
		case kr1 := <-resultDataChan:
			resultDataChanClose = true
			if kr1.error != nil {
				_logInfo(r, fmt.Sprintf("ResultData DeleteMulti error: %v", err))
				errors = append(errors, err)
			} else {
				if kr1.keys != nil && len(kr1.keys) > 0 {
					_logInfo(r, fmt.Sprintf("ResultData keys len: %d.", len(kr1.keys)))
					resultDataErrorChan = _deleteKeys(r, jsonKey, kr1.keys)
				} else {
					errorDataChanClose = true
				}
			}
		case kr2 := <-rateChan:
			rateChanClose = true
			if kr2.error != nil {
				_logInfo(r, fmt.Sprintf("Rate DeleteMulti error: %v", err))
				errors = append(errors, err)
			} else {
				if kr2.keys != nil && len(kr2.keys) > 0 {
					_logInfo(r, fmt.Sprintf("Rste keys len: %d.", len(kr2.keys)))
					rateErrorChan = _deleteKeys(r, jsonKey, kr2.keys)
				} else {
					errorChanClose = true
				}
			}
		}
	}

	if len(errors) > 0 {
		return errors
	}

	for {
		if errorDataChanClose && errorChanClose {
			break
		}
		select {
		case err := <-resultDataErrorChan:
			errorDataChanClose = true
			if err != nil {
				errors = append(errors, err)
			}
		case err := <-rateErrorChan:
			errorChanClose = true
			if err != nil {
				errors = append(errors, err)
			}
		}
	}

	return errors
}

func _logEror(r *http.Request, msg error) {
	if r != nil {
		ctx := appengine.NewContext(r)
		logAE.Errorf(ctx, "Error: %v.", msg)
	} else {
		log.Fatal(msg)
	}
}

func _logInfo(r *http.Request, msg string) {
	if r != nil {
		ctx := appengine.NewContext(r)
		logAE.Infof(ctx, "Info: %v", msg)
	} else {
		log.Printf("Info: %v", msg)
	}
}

func _getResultDataKeys(r *http.Request, jsonKey []byte, unixtime int64) <-chan keyResult {
	out := make(chan keyResult)
	go func() {
		defer close(out)
		var ctx context.Context
		if r != nil {
			ctx = appengine.NewContext(r)
		} else {
			ctx = context.Background()
		}

		client, err := datastore.NewClient(ctx, "rp-optima")
		if err != nil {
			out <- keyResult{error: err}
			return
		}

		if keys, err := client.GetAll(ctx, datastore.NewQuery("ResultData").Filter("timestamp<", unixtime).KeysOnly(), nil); err != nil {
			out <- keyResult{error: err}
		} else {
			out <- keyResult{keys: keys}
		}
	}()
	return out
}

func _getRatesKeys(r *http.Request, jsonKey []byte, unixtime int64) <-chan keyResult {
	out := make(chan keyResult)
	go func() {
		defer close(out)
		var ctx context.Context
		if r != nil {
			ctx = appengine.NewContext(r)
		} else {
			ctx = context.Background()
		}

		client, err := datastore.NewClient(ctx, "rp-optima")
		if err != nil {
			out <- keyResult{error: err}
			return
		}

		/*if keys, err := client.GetAll(ctx, datastore.NewQuery("ResultData").Filter("timestamp<", unixtime).KeysOnly(), nil); err != nil {
			out <- keyResult{error:err}
			return
		}else {
			out <- keyResult{keys:keys}
		}*/
		if keys, err := client.GetAll(ctx, datastore.NewQuery("Rate").Filter("id<", unixtime).KeysOnly(), nil); err != nil {
			out <- keyResult{error: err}
		} else {
			out <- keyResult{keys: keys}
		}
	}()
	return out
}

func _deleteKeys(r *http.Request, jsonKey []byte, keys []*datastore.Key) <-chan error {
	out := make(chan error)
	go func() {
		defer close(out)
		var ctx context.Context
		if r != nil {
			ctx = appengine.NewContext(r)
		} else {
			ctx = context.Background()
		}

		client, err := datastore.NewClient(ctx, "rp-optima")
		if err != nil {
			out <- err
			return
		}

		counter := len(keys) / 500
		for i := 0; i <= counter; i++ {
			low := i * 500
			top := len(keys) - 1
			if (i+1)*500 < top {
				top = (i + 1) * 500
			}
			if low > top {
				break
			}

			_logInfo(r, fmt.Sprintf("keys delete loop, first key: %v.", keys[low]))

			if err := client.DeleteMulti(ctx, keys[low:top]); err != nil {
				out <- err
				return
			}
		}

		out <- nil
	}()
	return out
}
