package main
import (
	"net/http"
//	"fmt"
//	"encoding/json"
//	"strconv"
//	"time"
//	"io/ioutil"
//	"pr.optima/src/core/entities"
	"pr.optima/src/server/rest/server"
//	"pr.optima/src/repository"
	"github.com/gorilla/mux"
)
var router *mux.Router

func init() {
	router = server.NewRouter()
	http.Handle("/api/", router)
}


