package controllers
import (
	"fmt"
	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
	"log"
	"net/http"
)

const historyLimit = 100

var (
	_initialized = false
	_rateRepo repository.RateRepo
	_rates []entities.Rate
// RUB
	_rubResultRepo repository.ResultDataRepo
	_rubEffRepo repository.EfficiencyRepo
	_rubResultList *entities.ResultDataListResponse
	_rubResult *entities.ResultDataResponse
	_rubSignal *entities.Signal
// EUR
	_eurResultRepo repository.ResultDataRepo
	_eurEffRepo repository.EfficiencyRepo
	_eurResultList *entities.ResultDataListResponse
	_eurResult *entities.ResultDataResponse
	_eurSignal *entities.Signal
// GBP
	_gbpResultRepo repository.ResultDataRepo
	_gbpEffRepo repository.EfficiencyRepo
	_gbpResultList *entities.ResultDataListResponse
	_gbpResult *entities.ResultDataResponse
	_gbpSignal *entities.Signal
// CHF
	_chfResultRepo repository.ResultDataRepo
	_chfEffRepo repository.EfficiencyRepo
	_chfResultList *entities.ResultDataListResponse
	_chfResult *entities.ResultDataResponse
	_chfSignal *entities.Signal
// CNY
	_cnyResultRepo repository.ResultDataRepo
	_cnyEffRepo repository.EfficiencyRepo
	_cnyResultList *entities.ResultDataListResponse
	_cnyResult *entities.ResultDataResponse
	_cnySignal *entities.Signal
// JPY
	_jpyResultRepo repository.ResultDataRepo
	_jpyEffRepo repository.EfficiencyRepo
	_jpyResultList *entities.ResultDataListResponse
	_jpyResult *entities.ResultDataResponse
	_jpySignal *entities.Signal
)

func initializeRepo(r *http.Request) {
	log.Print("repoInit start")
	_rateRepo = repository.New(historyLimit + 5, false, r)
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
	log.Print("repoInit end")
}

func reloadData(r *http.Request) {
	if _initialized == false {
		initializeRepo(r)
	}else {
		_rateRepo.Reload()
		_rates = _rateRepo.GetAll()

		_rubResultRepo.Reload()
		_rubEffRepo.Reload()

		_eurResultRepo.Reload()
		_eurEffRepo.Reload()

		_gbpResultRepo.Reload()
		_gbpEffRepo.Reload()

		_chfResultRepo.Reload()
		_chfEffRepo.Reload()

		_cnyResultRepo.Reload()
		_cnyEffRepo.Reload()

		_jpyResultRepo.Reload()
		_jpyEffRepo.Reload()
	}
	rebuildData()
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
		return nil, nil, nil, fmt.Errorf("PopulateSet error, in get last EFF not found.")
	}
	//	score10 := float32(eff.GetDirectionRate10())
	//	score100 := float32(eff.GetDirectionRate100())
	score10, score100 := get10_100Score(eff)
	//	if err != nil {
	//		return nil, nil, nil, fmt.Errorf("PopulateSet error, in get score: %v.", err)
	//	}
	results := resultRepo.GetAll()
	l := len(results)
	var resultSet []entities.ResultResponse
	limit := historyLimit
	if l < historyLimit {
		limit = l
	}
	resultSet = make([]entities.ResultResponse, limit)

	counter := 1
	for i := l - 1; i >= l - limit && i >= 0; i-- {
		result := new(entities.ResultResponse)
		result.Prediction = results[i].Prediction
		result.Result = results[i].Result
		result.Source = make([]int32, len(results[i].Source))
		copy(result.Source, results[i].Source)
		result.Timestamp = results[i].Timestamp

		if err := populateDataFor(result.Timestamp, results[i].Step, results[i].Symbol, results, result); err != nil {
			return nil, nil, nil, fmt.Errorf("PopulateSet error, in populate data: %v.", err)
		}
		result.Symbol = results[i].Symbol
		resultSet[limit - counter] = *result
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
	retCurrent.Data = *(resultSet[len(resultSet) - 1].Clone())
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

	var tenSum int32 = 0
	var hundredSum int32 = 0

	for i := 1; i <= index; i++ {
		item := eff.LastSD[l - i]
		if i <= 10 {
			tenSum += item
		}
		hundredSum += item
	}
	if index < 10 {
		return float32(tenSum) / float32(index), float32(hundredSum) / float32(index)
	}else {
		return float32(tenSum) / 10, float32(hundredSum) / float32(index)
	}
}

func populateDataFor(timestamp int64, frame int32, symbol string, results []entities.ResultData, result *entities.ResultResponse) error {
	l := len(_rates)
	addExtendedData := false
	if _rates[l - 1].Id > timestamp {
		addExtendedData = true
		frame++
	}
	result.Data = make([]float32, frame)
	result.Time = make([]int64, frame)

	var counter int32 = 1
	var prevRate entities.Rate
	for i := l - 1; i > 0 && counter <= frame; i-- {
		if _rates[i].Id <= timestamp {
			if addExtendedData {
				value, err := prevRate.GetForSymbol(symbol)
				if err != nil {
					return fmt.Errorf("ExtrudeDataFor error: %v", err)
				}
				result.Data[frame - counter] = value
				result.Time[frame - counter] = prevRate.Id
				counter++
				addExtendedData = false
			}
			value, err := _rates[i].GetForSymbol(symbol)
			if err != nil {
				return fmt.Errorf("ExtrudeDataFor error: %v", err)
			}
			result.Data[frame - counter] = value
			result.Time[frame - counter] = _rates[i].Id
			counter++
		}
		prevRate = _rates[i]
	}
	return nil
}