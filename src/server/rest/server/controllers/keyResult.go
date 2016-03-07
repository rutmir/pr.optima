package controllers
import "google.golang.org/cloud/datastore"

type keyResult struct{
	keys []*datastore.Key
	error error
}