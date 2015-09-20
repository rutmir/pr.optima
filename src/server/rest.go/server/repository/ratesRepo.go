package repository

import (
	"fmt"
	"../../entities"
)

var _limit int = 9
var _lastId int64
var _rates    []entities.Rate

func init() {
	/*RepoCreateTodo(entities.T_odo{Name: "Write presentation"})
	RepoCreateTodo(entities.T_odo{Name: "Host meetup"})*/
}

/*
func RepoFindTodo(id int) entities.T_odo {
	for _, t := range Todos {
		if t.Id == id {
			return t
		}
	}
	// return empty Todo_ if not found
	return entities.T_odo{}
}*/


func AppendNewRate(r entities.Rate) (entities.Rate, error) {
	if r.Id < _lastId + 3500 {
		return r, fmt.Errorf("Hour shift required (last: %d, new: %d).", _lastId, r.Id)
	}
	_lastId = r.Id
	_rates = append(_rates, r)
	var l = len(_rates)
	if l > _limit {
		_rates = _rates[l - _limit:]
	}
	fmt.Printf("len(_rates): %d", len(_rates))
	fmt.Println("")
	return r, nil
}

/*
func RepoDestroyTodo(id int) error {
	for i, t := range Todos {
		if t.Id == id {
			Todos = append(Todos[:i], Todos[i + 1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find T_odo with id of %d to delete", id)
}*/
