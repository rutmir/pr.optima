package jobs
import 	(
	"fmt"
	"log"
	"net/http"
	"encoding/json"

	"google.golang.org/appengine"
	"google.golang.org/appengine/urlfetch"
	logAE "google.golang.org/appengine/log"

	"pr.optima/src/core/entities"
	"pr.optima/src/repository"
	"pr.optima/src/grabber/work"
	"pr.optima/src/server/rest/server/controllers"
)

const (
	//source1Url = "https://openexchangerates.org/api/latest.json?app_id=7cb63de0a50c4a9e88954d825b6505a1&base=USD"
	source2Url = "http://www.apilayer.net/api/live?access_key=85c7d5e8f98fe83fa3fa81aafe489022&currencies=RUB,JPY,GBP,USD,EUR,CNY,CHF" //"85c7d5e8f98fe83fa3fa81aafe489022"
	//appEngineUrl = "https://rp-optima.appspot.com/api/refresh"
	repoSize = 200
)

//const authKey = "B7C05147C5A34376B30CEF2F289FBB6C"
var (
	//_repo repository.RateRepo
	works map[string] *fetchRatesWorkItem
)

func init() {
	works = make(map[string] *fetchRatesWorkItem)
	works ["RUB"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "RUB")
	works ["EUR"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "EUR")
	works ["GBP"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "GBP")
	works ["JPY"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "JPY")
	works ["CNY"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "CNY")
	works ["CHF"] = NewFetchRatesWorkItem(6, 5, 20, 1, work.TTLbfgs, "CHF")
}

func FetchRatesJob(w http.ResponseWriter, r *http.Request) {
	_, success, err := updateFromSource2ForAppEngine(r)
	if err != nil {
		if r != nil {
			ctx := appengine.NewContext(r)
			logAE.Errorf(ctx, "FetchRatesJob Fatal: %v", err)
			w.Header().Set("Cache-Control", "no-cache")
			http.Error(w, fmt.Sprintf( "FetchRatesJob Fatal: %v", err), http.StatusInternalServerError)
		}else {
			log.Fatal(err)
		}
		return
	}

	if success == false {
		if r != nil {
			ctx := appengine.NewContext(r)
			logAE.Errorf(ctx, "FetchRatesJob Error")
			w.Header().Set("Cache-Control", "no-cache")
			http.Error(w, fmt.Sprintf( "FetchRatesJob Error"), http.StatusInternalServerError)
		}
		return
	}
	executeDomainLogic(w, r)

}

func executeDomainLogic(w http.ResponseWriter, r *http.Request) {
	repo := repository.New(repoSize, true, r)
	rates := repo.GetAll()

	for key, work := range  works {
		if work.Limit < len(rates) {
			_, err := work.Process(rates, r)

			if err != nil {
				if r != nil {
					ctx := appengine.NewContext(r)
					logAE.Warningf(ctx, "Error: %v", fmt.Errorf("%s executeDomainLogic error: %v", key, err))
				}else {
					log.Printf("%s executeDomainLogic error: %v", key, err)
				}
			}
		}
	}

	controllers.ReloadData(r)

	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
}

func updateFromSource2ForAppEngine(r *http.Request) (int64, bool, error) {
	var result int64 = 0
	ctx := appengine.NewContext(r)
	client := urlfetch.Client(ctx)
	resp, err := client.Get(source2Url)
	if err != nil {
		return 0, false, err
	}
	defer resp.Body.Close()

	dec := json.NewDecoder(resp.Body)
	if resp.StatusCode == 200 {
		var rate entities.Rate2Response
		dec.Decode(&rate)
		result = rate.TimestampUnix
		//log.Println(rate.ToShortString())
		repo := repository.New(repoSize, true, r)
		if err := repo.Push(entities.Rate{
			Base    : rate.Base,
			Id      : rate.TimestampUnix,
			RUB     : rate.Quotes["USDRUB"],
			JPY     : rate.Quotes["USDJPY"],
			GBP     : rate.Quotes["USDGBP"],
			USD     : rate.Quotes["USDUSD"],
			EUR     : rate.Quotes["USDEUR"],
			CNY     : rate.Quotes["USDCNY"],
			CHF     : rate.Quotes["USDCHF"]}); err != nil {
			//log.Printf("Push rate to repo error: %v.", err)
			return 0, false, fmt.Errorf("Push rate to repo error: %v", err)
		}
	}else {
		var err2Resp entities.Error2Response
		dec.Decode(&err2Resp)
		return 0, false, fmt.Errorf(err2Resp.ToString())
	}
	return result, true, nil
}


