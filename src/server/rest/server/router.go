package server

import (
	"net/http"

	"github.com/gorilla/mux"
)

// NewRouter - return instance of main server router
func NewRouter() *mux.Router {

	router := mux.NewRouter().StrictSlash(true)

	for _, element := range routes {
		var handler http.Handler
		route := Route(element)
		handler = route.HandlerFunc
		handler = Logger(handler, route.Name)

		router.Methods(route.Method).Path(route.Pattern).Name(route.Name).Handler(handler)
	}
	return router
}
