package entities

import "fmt"

// Error2Response struct
type Error2Response struct {
	Success bool             `json:"success"`
	Error   ErrorDescription `json:"error"`
}

// ErrorDescription struct
type ErrorDescription struct {
	Code int    `json:"code"`
	Info string `json:"info"`
}

// ToString method
func (f *Error2Response) ToString() string {
	return fmt.Sprintf("Error - Code: %d, Info: %s", f.Error.Code, f.Error.Info)
}
