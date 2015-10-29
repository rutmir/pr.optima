package controllers

import (
	"fmt"
	"net/http"
	"github.com/gorilla/mux"
)

func Index(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Welcome!\n")
}

func Current(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	format := vars["format"]
	symbol := vars["symbol"]
	fmt.Fprintf(w, "Current { Symbol: %v, Format: %v }\n", symbol, format)
}