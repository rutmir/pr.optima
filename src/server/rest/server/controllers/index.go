package controllers

import (
	"fmt"
	"net/http"
)

// Index - handler for default request
func Index(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Welcome!\n")
}
