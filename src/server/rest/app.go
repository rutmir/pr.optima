package main

import (
	"net/http"

	"github.com/gorilla/mux"
	
	"pr.optima/src/server/rest/server"
)

var router *mux.Router

func init() {
	router = server.NewRouter()
	http.Handle("/", router)
}
