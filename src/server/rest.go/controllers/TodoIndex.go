package controllers

import (
	"encoding/json"
	"net/http"

	"../entities"
)

func TodoIndex(w http.ResponseWriter, r *http.Request) {
	todos := entities.Todos{
		entities.Todo{Name: "Write presentation"},
		entities.Todo{Name: "Host meetup"},
	}

	if err := json.NewEncoder(w).Encode(todos); err != nil {
		panic(err)
	}
}