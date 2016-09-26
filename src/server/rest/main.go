// +build !appengine
// appcfg.py -A rp-optima update app.yaml
// appcfg.py -A rp-optima update ./
package main
import (
	"log"
	"net/http"
)

func main() {
	log.Fatal(http.ListenAndServe(":8080", nil))
}