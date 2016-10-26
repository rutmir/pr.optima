package server

import (
	"net/http"

	"pr.optima/src/server/rest/server/controllers"
	"pr.optima/src/server/rest/server/jobs"
)

// Route - type holding routing information
type Route struct {
	Name        string
	Method      string
	Pattern     string
	HandlerFunc http.HandlerFunc
}

// Routes - list of known routes
type Routes []Route

var routes = Routes{
	Route{"Index", "GET", "/", controllers.Index},
	Route{"GetCurrentData", "GET", "/api/{format}/{symbol}/current", controllers.Current},
	Route{"GetAllData", "GET", "/api/{format}/{symbol}/all", controllers.All},
	Route{"GetAdvisor", "GET", "/api/{format}/{symbol}/advisor", controllers.Advisor},
	Route{"RefreshData", "GET", "/api/refresh", controllers.Refresh},
	Route{"CleanData", "GET", "/api/clean", controllers.ClearDB},
	Route{"FetchRates", "GET", "/jobs/fetch-rates", jobs.FetchRatesJob},
}
