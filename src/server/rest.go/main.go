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
	//	data := map[string]interface{}{}
	var data entities.Cources
	resp, err := http.Get("https://openexchangerates.org/api/historical/2015-09-08.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD")
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	//	body, err := ioutil.ReadAll(resp.Body)
	//	if err != nil{
	//		log.Fatal(err)
	//	}


	dec := json.NewDecoder(resp.Body)
	dec.Decode(&data)
	fmt.Print(data.ToString())
	//	log.Println(dec)

	//	json.Unmarshal(body, &data)
	//
	//	for key, val := range data {
	//		fmt.Print(key)
	//		fmt.Print(": ")
	//		fmt.Println(val)
	//	}

	router := server.NewRouter()
	log.Fatal(http.ListenAndServe(":8080", router))
}
