---
layout: post
title:  "First Steps In Go - Web Services"
date:   2021-01-12 09:15:16 +0200
categories: programming
---

These are my first steps in Go. This time, learning how to build web services in golang.  


### Listening to Incoming Requests

```golang
package main

import (
	"log"
	"net/http"
)

func customEndpoint(w http.ResponseWriter, r *http.Request) {
	w.Write([]bytes("Hello World"))
	log.Println("Served.")
}

func main() {

	// a /custom endpoint
	http.HandleFunc("/custom", customEndpoint)

	// listen on localhost, port :8080
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}

}
```

### Handling JSON

If we want to export a field from a structure to JSON, its name has to start with a capital letter, making it a public symbol. Otherwise it will be considered as private and it will not appear in the output string.

We use annotations, which are accessible at runtime through reflection, to specify how the field will be serialized. There is no space between `json`, `:` and the name. If we skip the annotation, the structure will be serialized with its fields as JSON fields.

```go
import "encoding/json"

type Product struct {
	ProductID      int    `json:"productId"`
	Manufacturer   string `json:"manufacturer"`
	PricePerUnit   string `json:"pricePerUnit"`
	UnitsAvailable int    `json:"unitsAvailable"`
	ProductName    string `json:"productName"`
}
```

To serialize JSON we do the following:

```go
if bytes, err := json.Marshal(&Product{
	ProductID:      0,
	Manufacturer:   "Apple",
	PricePerUnit:   "2500EUR",
	UnitsAvailable: 15,
	ProductName:    "MacBook Pro",
}); err == nil {
	log.Println("Successfully serialized to JSON")
} else {
	log.Println("Failed to serialize object")
}
```

To deserialize, the following:

```go
product := Product{}
err = json.Unmarshal(serializedJSONString, &product)

if err != nil {
	log.Println("Could not unmarshal")
}
```

### Handling of HTTP Verbs

A simple WebService, handling the GET method, returning a list of products from an in-memory structure.

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
)

// Product type
type Product struct {
	ProductID      int    `json:"productId"`
	Manufacturer   string `json:"manufacturer"`
	PricePerUnit   string `json:"pricePerUnit"`
	UnitsAvailable int    `json:"unitsAvailable"`
	ProductName    string `json:"productName"`
}

// some products stored in memory to play a bit
var products []*Product

// endpoint handler
func productsHandler(w http.ResponseWriter, r *http.Request) {

	switch r.Method {

	// handling the GET verb
	case http.MethodGet:
		jsonStr, err := json.Marshal(products)
		if err != nil {
			log.Println(err)
			w.WriteHeader(http.StatusInternalServerError)
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(jsonStr))
		}
	// everything else, not impemented
	default:
		w.WriteHeader(http.StatusNotImplemented)
	}
}

func main() {

	// init a few products in memory
	products = []*Product{}

	for i := 0; i < 10; i++ {
		products = append(products, &Product{
			ProductID:      i,
			Manufacturer:   "Apple",
			PricePerUnit:   fmt.Sprintf("%vEUR", (rand.Int()%10)*100+500),
			UnitsAvailable: rand.Int() % 15,
			ProductName:    "MacBook Pro",
		})
	}

	// handler
	http.HandleFunc("/products", productsHandler)

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
```

And the output:

![Basic Service Running]({{site.url}}/assets/gows1.png)

To create a new product, we update the switch block from above with the following:

```go
case http.MethodPost:
	body, err := ioutil.ReadAll(
		&io.LimitedReader{ // ensure we don't get DoS
			R: r.Body,
			N: 1024})

	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	product := Product{}
	err = json.Unmarshal(body, &product)

	if err != nil || product.ProductID != 0 {

		if err == nil {
			err = errors.New("ProductID should be 0 - if you know the ID, use PUT")
		}

		log.Println(err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// give them an increment
	// for now assume products are in incremental order, sorted
	// ensure safe to this data structure
	mtx.Lock()
	defer mtx.Unlock()

	if len(products) > 0 {
		product.ProductID = products[len(products)-1].ProductID + 1
	}

	products = append(products, &product)
	w.WriteHeader(http.StatusCreated)
```

To test the service we just do

```
$curl -D - -X POST -H "Content-Type: application/json" -d '{"productId" : 0, "manufacturer": "Microsoft", "productName": "MS Surface"}' localhost:8080/products
```

And we get the expected response back

```
HTTP/1.1 201 Created
Date: Sat, 16 Jan 2021 09:09:20 GMT
Content-Length: 0
```

What we are going to do now is to implement `GET` for a specific product ID and `PUT` for updating a specific ID. 

To do this, we need a new handler which we add to the main function. This will match the trailing `/`.

```golang
// handler for GET id and PUT id
http.HandleFunc("/products/", productHandler)
```

The URLs that will go to this handler take the form `http://localhost:8080/products/id`. We are also going to structure a bit better the handler, so the error handling is factored out of the main function.

```go
func productHandler(w http.ResponseWriter, r *http.Request) {

	retCode := func(w http.ResponseWriter, r *http.Request) int {

		pathSegments := strings.Split(r.URL.Path, "/products/")

		if len(pathSegments) != 2 {
			return http.StatusBadRequest
		}

		productID, err := strconv.Atoi(pathSegments[len(pathSegments)-1])

		if err != nil {
			return http.StatusBadRequest
		}

		product := findProductByID(productID)

		if product == nil {
			return http.StatusNotFound
		}

		switch r.Method {
		case http.MethodGet:

			mtx.RLock()
			defer mtx.RUnlock()

			jsonStr, err := json.Marshal(product)
			if err != nil {
				return http.StatusInternalServerError
			}

			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(jsonStr))

			return http.StatusOK

		case http.MethodPut:

			mtx.Lock()
			defer mtx.Unlock()

			body, err := ioutil.ReadAll(
				&io.LimitedReader{
					R: r.Body,
					N: 1024})

			if err != nil || json.Unmarshal(body, &product) != nil {
				return http.StatusBadRequest
			}

			// ensure ID stays the same
			product.ProductID = productID
			return http.StatusAccepted
		default:
			return http.StatusMethodNotAllowed
		}

	}(w, r)

	log.Println(r.Method, r.URL.Path)
	w.WriteHeader(retCode)

}
```

Since we store the products in an array in memory, the find function is as simple as it gets.

```go

var mtx sync.RWMutex

func findProductByID(id int) *Product {

	mtx.RLock()
	defer mtx.RUnlock()

	for _, p := range products {
		if p != nil && p.ProductID == id {
			return p
		}
	}
	return nil
}
```

We can test our code easily from the command line invoking

```
$curl -D - -X GET http://localhost:8080/products/2
```

and

```
$curl -D - -X PUT -H "Content-Type: application/json" -d '{"productId": 0, "manufacturer": "Microsoft", "productName": "MS Surface"}' localhost:8080/products/2
```

### Adding Middlewares

The http package allows for easy addition of middleware. Such middleware can do things like authentication, caching (memoizing), logging or session management. We will modify our code to add a CORS middleware

```go
func corsMiddleware(handler http.Handler) http.Handler {

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		// before the handler
		// add the cors middleware headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length")
		w.Header().Set("Content-Type", "application/json")

		if r.Method == http.MethodOptions {
			// the pre-flight request, make sure it is handled
			return
		}

		// the actual handler
		handler.ServeHTTP(w, r)

		// after handler
	})
}


func main() {

	// [missing some code]

	// handler for GET all and POST
	http.Handle("/products", corsMiddleware(http.HandlerFunc(productsHandler)))

	// handler for GET id and PUT
	http.Handle("/products/", corsMiddleware(http.HandlerFunc(productHandler)))

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
```

The full code, refactored with persitence in an in-memory map can be found [here](https://github.com/alexandrugris/learngolang1/tree/persistence_in_memory_map)

