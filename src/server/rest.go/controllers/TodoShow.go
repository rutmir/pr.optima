package controllers

import (
	"net/http"
	"fmt"
	"github.com/gorilla/mux"
)

func TodoShow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	todoId := vars["todoId"]
	fmt.Fprintln(w, "Todo show:", todoId)
}
