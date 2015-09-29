package controllers

import (
	"encoding/json"
	"net/http"

	"rest.go/server/repository"
)

func TodoIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(repository.Todos); err != nil {
		panic(err)
	}
}