package server

import (
	"net/http"
	"pr.optima/src/server/rest/server/controllers"
)

type Route struct {
	Name        string
	Method      string
	Pattern     string
	HandlerFunc http.HandlerFunc
}

type Routes []Route

var routes = Routes{
	Route{"Index", "GET", "/", controllers.Index, },
	Route{"TodoIndex", "GET", "/todos", controllers.TodoIndex, },
	Route{"TodoCreate", "POST", "/todos", controllers.TodoCreate, },
	Route{"TodoShow", "GET", "/todos/{todoId}", controllers.TodoShow, },
	Route{"GetCurrentData", "GET", "/api/{format}/{symbol}/current", controllers.Current, },
}
