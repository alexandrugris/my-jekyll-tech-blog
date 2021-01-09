---
layout: post
title:  "First Steps In Go"
date:   2021-01-10 09:15:16 +0200
categories: programming
---

My first steps in Go, largely based on the Golang tutorial and side Internet searches. 

### Hello World

Building a very basic hello world project once the go tools are installed is straight forward:

 - Create a folder named "helloworld"
 - `cd ./helloworld`
 - Create a file called `main.go`
 - Add the following code into the file
 - Save and at the command prompt type `go build`. An executable file called `helloworld` will be compiled into the folder.

 ```go
package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello World")
}
 ```

A good editor for `go` is Visual Studio Code. 

### Language Basics

The most basic unit of organizing code in go is the function. Below is an example of a function with several parameters, one of each being a callback (function pointer). 

```go
// function receiving function as a parameter
func arrayOpScalar(array []float32, constant float32, operation func(float32, float32) float32) {
	for i := range array {
		array[i] = operation(array[i], constant)
	}
}
```

A function can be passed as a parameter, assigned to a variable or returned from another function.

```go
// use the arrayOpScalar function defined above to 
// create another function to double the values in an array
doubleFn := func(array []float32) {
		arrayOpScalar(array, 2, func(x, y float32) float32 { return x * y })
	}

doubleFn(p)
```

But before we do that, let's look a bit at arrays and slices. Allocating an array goes like this

```go
primes := [6]int{2, 3, 5, 7, 11, 13}
```

The length is part of the array so it cannot be resized. Slices are views onto arrays, so when a value is modified on the slice it will automatically propagate to the backing array.

```go
var s []int = primes[1:4]
```

The internal structure of a slice is as follows: 

```go
type slice struct {
  array *T,
  len int,
  cap int,
}
```
Length and capacity can be accessed through `len()` and `cap()`. Therefore, in golang you can do very cool stuff such as converting from a struct to its underlying byte representation. Such operations are useful when, for instance, memory mapping files to arrays of a specified stucture without additional serialization / deseralization. [Here for an extended thread](https://stackoverflow.com/questions/16330490/in-go-how-can-i-convert-a-struct-to-a-byte-array)

```go
type Struct struct {
    p1 int32
    p2 int32
    p3 uint16
    p3 uint16
}

// read in a compile-time constant the size of the struct
const sz = int(unsafe.SizeOf(Struct{}))

// initialize convert the pointer to the struct to an array of bytes 
// of the same size as the struct and take a slice to it.
var asByteSlice []byte = (*(*[sz]byte)(unsafe.Pointer(&struct_value)))[:]
```

Slices can contain other slices.

```go
mat3x3 := [][]float32{
		[]float32{1.0, 0.0, 0.0},
		[]float32{0.0, 1.0, 0.0},
		[]float32{0.0, 0.0, 1.0},
	}
// elements can be accessed
fmt.Println(board[0][0])
```

```go
  // dynamic allocation of an array of 10 floats
	p := make([]float32, 10)

  // dynamically growing the array by appending
  // 10 elements with spread operator
	p = append(p, make([]float32, 10)...)

	// looping over indexes in p
	for i := range p {
		p[i] = float32(i)
	}
```

Let's look also at static initialization.

```go
slice := []struct { // annonymous struct of two integers
		i1 int
		i2 int
	}{ // statically initialized
		{0, 0},
		{1, 1},
		{2, 2}, // comma at the end is mandatory
	}

for _, x := range slice {
	fmt.Printf("%v : %v\n", x.i1, x.i2)
}
```

### Custom Types

In `go`, encapsulation is defined at the package level. Everything in a package is public. Exported symbols start with capital letter, everything starting with lowercase is private outside of the package. Let's define a custom type.

```go
type Vertex struct {
	X float64
	Y float64
	Z float64
}
```

Initializing a variable of such a type goes like this:

```go
v := Vertex{0.1, 0.2, 0.3}
```

or 

```go
v := Vertex {
X : 0.1,
Y : 0.2,
Z : 0.3,
}
```

We can return a pointer of such a struct. By default the compiler will favor stack allocation, but it does perform escape analysis and, in case the lifetime of an object cannot be determined at compile time it will switch to allocating it on the heap.

```go
func returnPointerToVertex() *Vertex {
	return &Vertex{1.0, 2.0, 3.0}
}
```

Now let's add some methods to the type. 

```go
// Length computes the vector norm
func (v Vertex) Length() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y + v.Z*v.Z)
}
```

Methods with pointer receivers can modify the value to which the receiver points (as scale does here). ince methods often need to modify their receiver, pointer receivers are more common than value receivers. There are two reasons to use a pointer receiver:
- The first is so that the method can modify the value that its receiver points to.
- The second is to avoid copying the value on each method call

In general, all methods on a given type should have either value or pointer receivers, but not a mixture of both.

```go
// Scale scales the vector by a float
func (v *Vertex) Scale(s float64) {

  // unlike C++ where invoking a method on a nullptr usually results in a crash
  // in golang this is perfectly acceptable
	if v == nil {
		fmt.Println("Received nill pointer")
		return
	}

	v.X *= s
	v.Y *= s
	v.Z *= s
}
```

Speaking of types, golang does not allow inheritance, but it does have the interface type. 

```go
type Scaler interface {
  Scale(float64)
}
```

Vector automatically implements this interface by simply implementing the respective methods. Now we can do

```go
v := Vertex{X: 0.1, Y: 0.2, Z: 0.3}
	
var scaler Scaler = &v
scaler.Scale(10.0)
```

Beside interfaces that have functions, go offers a very interesting concept, the empty interface. Any object can be assigned to the empty interface, including the scalar types. Here is an example:

```go
// empty interface
var intf interface{} = "Hello World"

// querying the empty interface for the underlying type
if s, ok := intf.(string); ok {
	fmt.Println(s)
}

// i := intf(float32) would panic
// need to test of OK
if i, ok := intf.(float32); ok {
	fmt.Println(i)
}

// a better way is to test with a type switch
// interesting is that v is the value converted to the type, not the type
switch v := intf.(type) {
	case string:
		fmt.Println("It's a string!", v)
	case int:
		fmt.Println("It's an int!", v)
	case float32:
		fmt.Println("It's a float!", v)
}
```

Speaking of the `switch` construct, it is quite flexible:

```go
// switch
// with declaration and condition
switch os := runtime.GOOS; os {
	case "linux":
		fallthrough
	case "windows", "darwin":
		fmt.Printf("Running on %v\n", os)
	default:
		fmt.Println("Unknown")
}

// with no condition
switch {
	case time.Now().Weekday().String() == "Thursday":
		fmt.Println("Today is Thursday")
	default:
		fmt.Println("Today is not Thursday")
}
```

### Maps

Maps can be initialized as literals or created dinamically with `make`

```go

// dynamic instantiation
m := make(map[string]Vertex, 10)
m["Iasi"] = Vertex{1.0, 1.0, 1.0}

// check for existence of an element
if _, exists := m["Cluj"]; !exists {
	fmt.Println("Cluj does not exist in the map")
}

fmt.Println(m["Iasi"].Length())

// literal instantiation
m1 := map[string]Vertex{
		"Iasi":      {1.0, 1.0, 1.0}, // no need to specify Vertex
		"Bucharest": {2.0, 2.0, 2.0},
}

// map can be increased
m1["Cluj"] = Vertex{3.0, 3.0, 3.0}

fmt.Println(m1)

// remove the element from the map
delete(m1, "Cluj")

// or also literal instantiation but with no elements
counts := map[string]int{}
```

### Sample Programs

Fibonnaci - function returning a function

```go
import "fmt"

// fibonacci is a function that returns
// a function that returns an int.
func fibonacci() func() int {
  
  // declaration - initialization
	first, second := 0, 1

	return func() int {
    ret := first + second
		first, second = second, ret
		return ret
	}
	
}

func main() {
	f := fibonacci()
	for i := 0; i < 10; i++ {
		fmt.Println(f())
	}
}
```

Error management and the Error interface:

```go
package main

import (
	"fmt"
	"math"
)

type ErrNegativeSqrt float64

func (v ErrNegativeSqrt) Error() string {
	if v < 0.0 {
		return fmt.Sprintf("Negative sqrt %v", float64(v))
	}
	return ""
}

func Sqrt(x float64) (float64, error) {
	
	if x < 0.0{
		return 0.0, ErrNegativeSqrt(x)
	}
	
	z := 1.0
	delta := z * z - x
	
	for math.Abs(delta) > 1e-10{ 
		z -= delta / (2.0 * z)
		delta = z * z - x 
	}
	
	return z, nil
}

func main() {
	if v, err := Sqrt(-2); err == nil {
		fmt.Println(v)	
	} else {
		fmt.Println(err)	
	}	
}
```

Reader implementation. An in-memory stream obtained from a string can be created with `r := strings.NewReader("Hello, Reader!")`

```go
package main

import (
	"io"
	"os"
	"strings"
)

type rot13Reader struct {
	r io.Reader
}

func (r rot13Reader) Read(b []byte) (int, error){
  
  // returns the number of elements read 
  // and an error if an error occured
  // the error can be io.EOF which signifies the end of the stream
	n, err := r.r.Read(b)
	
	for i := 0; i < n; i++{
		switch {
		case b[i] >= 'A' && b[i] <= 'Z': 
			b[i] = (b[i] - 'A' + 13) % 26 + 'A'
		case b[i] >= 'a' && b[i] <= 'z':
			b[i] = (b[i] - 'a' + 13) % 26 + 'a'
		}
	}
	
	return n, err 
}

func main() {
	s := strings.NewReader("Lbh penpxrq gur pbqr!")
	r := rot13Reader{s}
	io.Copy(os.Stdout, &r)
}
```

### Concurrency

Concurrency in go is achieved through goroutines. Goroutines are language constructs which maps M virtual threads to N CPU threads. The runtime has its own scheduler. The preferred way of of sharing state is through channels, although shared memory is also possible thanks to the `sync` standard package. Let's look at two programs below.

The first program compares two BSTs.

```go
package main

import (
	"golang.org/x/tour/tree"
	"fmt"
)

// Walk DFSes the tree t sending all values
// to the channel ch.
func Walk_(t *tree.Tree, ch chan int){

	if t.Left != nil{
		Walk_(t.Left, ch)
	}

  // send the current value to the channel
	ch <- t.Value

	if t.Right != nil {
		Walk_(t.Right, ch)
	}
}

func Walk(t *tree.Tree, ch chan int){
  Walk_(t, ch)

  // close the channel to signal the end of the tree
	close(ch) 
}

// Same determines whether the trees
// t1 and t2 contain the same values.

func Same(t1, t2 *tree.Tree) bool{

  // make two channels
	c1 := make(chan int)
	c2 := make(chan int)

  // launch the two walks in parallel
	go Walk(t1, c1)
	go Walk(t2, c2)

  // Read one value at a time from each channel
  // and compare them
	for ok1, ok2 := true, true; ok1 && ok2;  {
		var v1, v2 int

    // when one channel is closed, its OK value is set to false
		v1, ok1 = <- c1
		v2, ok2 = <- c2

		if ok1 != ok2 || v1 != v2{
			return false
		}

	}
	return true
}

func main() {
	if Same(tree.New(1), tree.New(1)) {
		fmt.Println("Same tree")
	} else {
		fmt.Println("Not the same tree")
	}
}
```

*Notes:*

 - A channel cannot be closed twice
 - A write from a closed channel results in a panic
 - You can check on read if the channel is closed
 - Channel operations are blocking. A channel can have a buffer, in which condition the operation becomes blocking only when the buffer is full
 - A channel can be read with a `range` construct. The range finishes when the sender closes the channel 

The second program, also part of the golang tour, introduces `sync.WaitGroups` to allow waiting for goroutines to finish as well as sending return channels through input channels for safe reply. To allow for concurrent access, the `Cache` is implemented as a process (actor) which is accessible only through its input and output channels. The code is commented extensively.

```go
package main

import (
	"fmt"
	"sync"
)

// the wait group is needed to allow all goroutines to signal when they finish execution
// and the main goroutine to wait for them
var wg sync.WaitGroup

type Fetcher interface {
	Fetch(url string) (body string, urls []string, err error)
}

// We are going to send a pair to our cache service
// <string - key, return channel>
// The return channel solves a concurrency issue:
// assuming that we have more concurrent readers waiting.
// we want to ensure we return the result to the reader that sent the message.
// Since in our case we use a non-buffered write channel, all writes are blocked until a 
// new read is performed and, since the cache is a single threaded, it will not make a new read until the 
// result is communicated, we could have used a single return channel for all the cache requests.
// however, if we make the write channel buffered, allowing for multiple writes, the returns will be mixed.
type CacheMsg struct {
	str string
	out chan bool
}

type Cache struct {
	in chan CacheMsg
}

func (p *Cache) Init(){
	p.in = make(chan CacheMsg)
	go p.cache()
}

func (p *Cache) Test(s string) bool {
  
  // create a new return channel for each service request
	msg := CacheMsg {
		str: s,
		out: make(chan bool),
	}
	
	p.in <- msg
	return <- msg.out
}

func (p *Cache) cache(){
  
  // our cache map
	cache := make(map[string]bool)
  
  // read messages with range until the channel is closed
	for msg := range p.in {
		if _, exists := cache[msg.str]; exists {
			msg.out <- true
		} else {
			cache[msg.str] = true
			msg.out <- false
		}
	}
}


func Crawl(url string, depth int, fetcher Fetcher, cache *Cache) {

  // ensure we call wg.Done() when the method exits
  defer wg.Done()

	if depth <= 0 {
		return
	}
  
  // the url is already in the cache
	if cache.Test(url) {
		return
	}
	
	body, urls, err := fetcher.Fetch(url)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("found: %s %q\n", url, body)
  
  // add N new goroutines to the WaitGroup
	wg.Add(len(urls))
	for _, u := range urls {

    // launch crawl goroutines in parallel
		go Crawl(u, depth-1, fetcher, cache)
	}
	return
}

func main() {
	
	var cache Cache
	cache.Init()
	
	wg.Add(1)
  Crawl("https://golang.org/", 4, fetcher, &cache)
  // wait for all goroutines to finish
	wg.Wait()
}

// fakeFetcher is Fetcher that returns canned results.
type fakeFetcher map[string]*fakeResult

type fakeResult struct {
	body string
	urls []string
}

func (f fakeFetcher) Fetch(url string) (string, []string, error) {
	if res, ok := f[url]; ok {
		return res.body, res.urls, nil
	}
	return "", nil, fmt.Errorf("not found: %s", url)
}

// fetcher is a populated fakeFetcher.
var fetcher = fakeFetcher{
	"https://golang.org/": &fakeResult{
		"The Go Programming Language",
		[]string{
			"https://golang.org/pkg/",
			"https://golang.org/cmd/",
		},
	},
	"https://golang.org/pkg/": &fakeResult{
		"Packages",
		[]string{
			"https://golang.org/",
			"https://golang.org/cmd/",
			"https://golang.org/pkg/fmt/",
			"https://golang.org/pkg/os/",
		},
	},
	"https://golang.org/pkg/fmt/": &fakeResult{
		"Package fmt",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
	"https://golang.org/pkg/os/": &fakeResult{
		"Package os",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
}
```
The implementation above is more generic as it can be used as a pattern for other kinds of services. In our case, a faster solution would have been to use shared memory protected through a `sync.Mutex`, `sync.RWMutex` or through a `sync.Map`, a concurrent map. 

One thing to note - altough all IO operations in go are blocking the current goroutine, the are implemented as asyncio behind the scenes, in a similar manner to which the `cache.Test()` method above is blocking.

### Timers and select

Select allows to listen to multiple channels and block until one of them has data available. Timers in golang are implemented as channels. Signaling to a goroutine to finish its job can be done also though a channel.

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

func main() {

	seconds := time.NewTicker(time.Second)
	minutes := time.NewTicker(time.Minute)

	done := make(chan bool)

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		for {
			select {
			case <-done:
				wg.Done()
				return // exit the routine
			case <-seconds.C:
				fmt.Println("Tick")
			case <-minutes.C:
				fmt.Println("Tock")
			}
		}
	}() // immediately invoked goroutine

	time.Sleep(time.Minute * 3)
	done <- true

	wg.Wait()
	fmt.Println("Done.")
}
```

### Conclusion

Go is a very beautiful and performant language. It is low level enough to feel like you have power you have in C and it compiles to native code for super fast startup times, performance and interoperability. It is elegant as it does not have unnecessary constructs yet, though its constructs, it encourages at the language level clean code and excellent concurrency. 





