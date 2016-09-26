package entities
import "fmt"

type Error2Response struct {
	Success bool              `json:"success"`
	Error   ErrorDescription  `json:"error"`
}

type ErrorDescription struct {
	Code int     `json:"code"`
	Info string  `json:"info"`
}

func (f *Error2Response) ToString() string {
	return fmt.Sprintf("Error - Code: %d, Info: %s", f.Error.Code, f.Error.Info)
}