package repository

import (
	"fmt"
	"rest.go/entities"
)

var currentId int
var Todos entities.Todos

// Give us some seed data
func init() {
	RepoCreateTodo(entities.Todo{Name: "Write presentation"})
	RepoCreateTodo(entities.Todo{Name: "Host meetup"})
}

func RepoFindTodo(id int) entities.Todo {
	for _, t := range Todos {
		if t.Id == id {
			return t
		}
	}
	// return empty Todo_ if not found
	return entities.Todo{}
}

//this is bad, I don't think it passes race condtions
func RepoCreateTodo(t entities.Todo) entities.Todo {
	currentId += 1
	t.Id = currentId
	Todos = append(Todos, t)
	return t
}

func RepoDestroyTodo(id int) error {
	for i, t := range Todos {
		if t.Id == id {
			Todos = append(Todos[:i], Todos[i + 1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find Todo with id of %d to delete", id)
}