package main

import (
	"log"
	"net/http"
	"./server"
//	"io/ioutil"
	"encoding/json"
	"fmt"
	"./entities"
)

func main() {
	var _cours entities.Cources
	var _error entities.Error

	resp, err := http.Get("https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD")
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	dec := json.NewDecoder(resp.Body)

	if resp.StatusCode == 200 {
		dec.Decode(&_cours)
		fmt.Print(_cours.ToString())
	} else {
		dec.Decode(&_error)
		fmt.Print(_error.ToString())
	}

	router := server.NewRouter()
	log.Fatal(http.ListenAndServe(":8080", router))
}
