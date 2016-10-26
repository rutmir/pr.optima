package controllers

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func writeError(w http.ResponseWriter, error string, code int) {
	// allow cross domain AJAX requests
	w.Header().Add("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "no-cache")

	http.Error(w, error, code)
}

func writeErrorJSON(w http.ResponseWriter, code int, i interface{}) {
	w.Header().Add("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "no-cache")
	buf, err := json.Marshal(i)
	if err != nil {
		http.Error(w, fmt.Sprintf("json.Marshal failed: %v", err), code)
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		w.Write(buf)
	}
}
