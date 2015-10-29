package controllers
import (
	"fmt"
	"strings"
	"net/http"
	"github.com/gorilla/mux"

	"pr.optima/src/core/entities"
)

const (
	_json operationFormat = iota
	_xml
	_protoBuf
	_text
)

type operationFormat int

func Current(w http.ResponseWriter, r *http.Request) {
	var format operationFormat

	vars := mux.Vars(r)
	f := strings.ToLower(vars["format"])

	switch f {
	case "json":
		format = _json
	case "protobuf":
		format = _protoBuf
	case "text":
		format = _text
	default:
		writeError(w, "Wrong return format. Required 'json', 'protobuf' or 'text'", http.StatusBadRequest)
		return
	}

	symbol := strings.ToUpper(vars["symbol"])

	switch symbol {
	case "RUB":
		fmt.Fprintf(w, "Current { Symbol: %v, Format: %v }\n", symbol, f)
		return
	}

	switch format {
	case _json:
		writeErrorJSON(w, http.StatusBadRequest, entities.ErrorResponse{Error : true, Status : http.StatusBadRequest, Message: fmt.Sprintf("Symbol: %v not supported.", symbol) })
	default:
		writeError(w, fmt.Sprintf("Symbol: %v not supported.", symbol), http.StatusBadRequest)
	}
}