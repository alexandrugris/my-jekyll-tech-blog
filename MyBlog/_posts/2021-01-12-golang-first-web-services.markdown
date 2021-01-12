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
			log.Fatal(err)
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


