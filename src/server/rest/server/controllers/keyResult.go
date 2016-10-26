package controllers

import "cloud.google.com/go/datastore"

type keyResult struct {
	keys  []*datastore.Key
	error error
}
