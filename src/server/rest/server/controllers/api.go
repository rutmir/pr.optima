package controllers
import (
	"fmt"
	"strings"
	"net/http"
	"encoding/json"
	"github.com/gorilla/mux"

	"pr.optima/src/core/entities"
	"log"
	"time"
)

type operationFormat int

const (
	_json operationFormat = iota
	_xml
	_protoBuf
	_text
)
const _authKey = "B7C05147C5A34376B30CEF2F289FBB6C"
var _supportedSymbols = []string{"RUB", "EUR", "GBP", "CHF", "CNY", "JPY"}

func Current(w http.ResponseWriter, r *http.Request) {
	format, symbol, found := processFormatAndSymbol(w, r)
	if found == false {
		return
	}

	switch symbol {
	case "RUB":
		returnCurrent(w, format, symbol, _rubResult)
		return
	case "EUR":
		returnCurrent(w, format, symbol, _eurResult)
		return
	case "GBP":
		returnCurrent(w, format, symbol, _gbpResult)
		return
	case "CHF":
		returnCurrent(w, format, symbol, _chfResult)
		return
	case "CNY":
		returnCurrent(w, format, symbol, _cnyResult)
		return
	case "JPY":
		returnCurrent(w, format, symbol, _jpyResult)
		return
	}
	returnError(w, fmt.Sprintf("Symbol: %v not implmented.", symbol), http.StatusNotImplemented, format)
}

func All(w http.ResponseWriter, r *http.Request) {
	format, symbol, found := processFormatAndSymbol(w, r)
	if found == false {
		return
	}

	switch symbol {
	case "RUB":
		returnResult(w, _rubResultList, format)
		return
	case "EUR":
		returnResult(w, _eurResultList, format)
		return
	case "GBP":
		returnResult(w, _gbpResultList, format)
		return
	case "CHF":
		returnResult(w, _chfResultList, format)
		return
	case "CNY":
		returnResult(w, _cnyResultList, format)
		return
	case "JPY":
		returnResult(w, _jpyResultList, format)
		return
	}
	returnError(w, fmt.Sprintf("Symbol: %v not implmented.", symbol), http.StatusNotImplemented, format)
}

func Advisor(w http.ResponseWriter, r *http.Request) {
	format, symbol, found := processFormatAndSymbol(w, r)
	if found == false {
		return
	}
	switch symbol{
	case "RUB":
		returnAdvisor01(w, format, symbol, _rubSignal)
		return
	case "EUR":
		returnAdvisor01(w, format, symbol, _eurSignal)
		return
	case "GBP":
		returnAdvisor01(w, format, symbol, _gbpSignal)
		return
	case "CHF":
		returnAdvisor01(w, format, symbol, _chfSignal)
		return
	case "CNY":
		returnAdvisor01(w, format, symbol, _cnySignal)
		return
	case "JPY":
		returnAdvisor01(w, format, symbol, _jpySignal)
		return
	}
	returnError(w, fmt.Sprintf("Symbol: %v not implmented.", symbol), http.StatusNotImplemented, format)
}

func Refresh(w http.ResponseWriter, r *http.Request) {
	authKey := r.Header.Get("Auth")
	if authKey != _authKey {
		returnError(w, "Request not authorized", http.StatusUnauthorized, _text)
		return
	}
	ReloadData(r)
	returnResult(w, "success", _text)
}

func ClearDB(w http.ResponseWriter, r *http.Request) {
	authKey := r.Header.Get("Auth")
	if authKey != _authKey {
		returnError(w, "Request not authorized", http.StatusUnauthorized, _text)
		return
	}
	errors := clearDbData(time.Now().Add(time.Hour * 24 * (-31)).UTC().Unix(), r)
	if errors == nil || len(errors) < 1 {
		returnResult(w, "success", _text)
	}else {
		errStr := "Errors: \n"
		for err := range errors {
			errStr += fmt.Sprintf("%v\n", err)
		}
		returnError(w, errStr, http.StatusInternalServerError, _text)
	}
}

func ReloadData(r *http.Request) {
	initializeRepo(r)
	rebuildData()
}

func returnCurrent(w http.ResponseWriter, format operationFormat, symbol string, set *entities.ResultDataResponse) {
	if set != nil {
		log.Println("returnCurrent 1")
		returnResult(w, set, format)
	}else {
		log.Println("returnCurrent 2")
		returnError(w, fmt.Sprintf("Data not exist for symbol: %s.", symbol), http.StatusBadRequest, format)
	}
}

func returnAdvisor01(w http.ResponseWriter, format operationFormat, symbol string, signal *entities.Signal) {
	if signal != nil {
		returnResult(w, *signal, format)
	}else {
		returnError(w, fmt.Sprintf("Data not exist for symbol: %s.", symbol), http.StatusBadRequest, format)
	}
}

func returnError(w http.ResponseWriter, err string, code int, format operationFormat) {
	switch format {
	case _json:
		writeErrorJSON(w, code, entities.ErrorResponse{Error : true, Status : code, Message: err })
	default:
		writeError(w, err, code)
	}
}

func returnResult(w http.ResponseWriter, obj interface{}, format operationFormat) {
	switch format {
	case _json:
		returnJson(w, obj)
	default:
		w.Header().Add("Access-Control-Allow-Origin", "*")
		//w.Header().Set("Cache-Control", "max-age=600, must-revalidate")
		w.Header().Set("Cache-Control", "no-cache")
		fmt.Fprintln(w, obj)
	}
}

func returnJson(w http.ResponseWriter, obj interface{}) {
	w.Header().Add("Access-Control-Allow-Origin", "*")
	buf, err := json.Marshal(obj)
	if err != nil {
		w.Header().Set("Cache-Control", "no-cache")
		http.Error(w, fmt.Sprintf("json.Marshal failed: %v", err), http.StatusInternalServerError)
	}else {
		w.Header().Set("Content-Type", "application/json")
		//w.Header().Set("Cache-Control", "max-age=600, must-revalidate")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		w.Write(buf)
	}
}

func processFormatAndSymbol(w http.ResponseWriter, r *http.Request) (operationFormat, string, bool) {
	var format operationFormat
	vars := mux.Vars(r)
	symbol := strings.ToUpper(vars["symbol"])

	switch strings.ToLower(vars["format"]) {
	case "json":
		format = _json
	case "protobuf":
		format = _protoBuf
	case "text":
		format = _text
	default:
		returnError(w, "Wrong return format. Required 'json', 'protobuf' or 'text'", http.StatusBadRequest, _text)
		return _text, symbol, false
	}

	if stringInSlice(symbol, _supportedSymbols) {
		return format, symbol, true
	}
	returnError(w, fmt.Sprintf("Symbol: %v not supported.", symbol), http.StatusBadRequest, format)
	return _text, symbol, false
}

func getSymbol(r *http.Request) (string, error) {
	vars := mux.Vars(r)
	symbol := strings.ToUpper(vars["symbol"])

	if stringInSlice(symbol, _supportedSymbols) {
		return symbol, nil
	}
	return symbol, fmt.Errorf("Symbol: %v not supported.", symbol)
}

func stringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}