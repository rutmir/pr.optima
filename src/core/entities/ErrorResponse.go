package entities

import "fmt"

// ErrorResponse struct
type ErrorResponse struct {
	Error       bool   `json:"error"`
	Status      int    `json:"status"`
	Message     string `json:"message"`
	Description string `json:"description"`
}

// ToString method
func (f *ErrorResponse) ToString() string {
	return fmt.Sprintf("Error: %t\nStatus: %d\nMessage: %s\nDescription: %s", f.Error, f.Status, f.Message, f.Description)
}
