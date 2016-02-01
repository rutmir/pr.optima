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
	Route{"GetCurrentData", "GET", "/api/{format}/{symbol}/current", controllers.Current, },
	Route{"GetAllData", "GET", "/api/{format}/{symbol}/all", controllers.All, },
	Route{"GetAdvisor", "GET", "/api/{format}/{symbol}/advisor", controllers.Advisor, },
	Route{"RefreshData", "GET", "/api/refresh", controllers.Refresh, },
	Route{"CleanData", "GET", "/api/clean", controllers.ClearDB, },
}
