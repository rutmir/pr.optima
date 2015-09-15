package entities

import "fmt"

type Error struct {
	Error       bool	`json:"error"`
	Status      int		`json:"status"`
	Message     string	`json:"message"`
	Description string	`json:"description"`
}

func (f Error) ToString() string {
	return fmt.Sprintf("Error: %t\nStatus: %d\nMessage: %s\nDescription: %s", f.Error, f.Status, f.Message, f.Description)
}