+++
title = "Learning Go"
description = "Simple book on playing with Go"
+++
# 1 Setup

``` terminal
# environment; defined in .bash_aliases.sh
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# code location
cd /Users/chang/Documents/dev/git/gratia/01_languages/goLang/book_learning_go
cd ch1
go run hello.go

go build -o hello_world hello.go

# install 3rd party tool to do a load test
go install github.com/rakyll/hey@latest
hey https://www.golang.org

# install 3rd party too to format imports
go install golang.org/x/tools/cmd/goimports@latest

# detect shadow variables
go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow@latest

```


- go run
    - use when you want to run as a script
    - compiles and put binary in a temporary directory

- go build -o [BINARY_NAME] hello.go
    - outputs the binary

- fetch 3rd party tools
    - Go’s method for publishing code is a bit different than most other languages. Go developers don’t rely on a centrally hosted service, like Maven Central for Java or the NPM registry for JavaScript. Instead, they share projects via their source code repositories

- autoformatting
    -  formats the file
        - go fmt .

    - format the imports
        - goimports -l -w .
            - the `-l` flag tells `goimports` to print the files with incorrect formatting to the console. The `-w` flag tells `goimports` to modify the files in-place. The `.` specifies the files to be scanned

- linting
    - makes the code follows a style per
        - [effective go](https://go.dev/doc/effective_go)
        - [go review](https://github.com/golang/go/wiki/CodeReviewComments)
    - setup
        - go install golang.org/x/lint/golint@latest
        - run from top of the current project
            - golint ./...
- makefile
    - makefile to fmt, lint, and build
```
.DEFAULT_GOAL := build

fmt:
        go fmt ./...
.PHONY:fmt

lint: fmt
        golint ./...
.PHONY:lint

vet: fmt
        go vet ./...
.PHONY:vet

build: vet
        go build hello.go
.PHONY:build
```

*  make ch1 a go module, which is needd for
    * cd ch1 && go mod init ch1
    * what is a go module?
      ```Go modules commonly consist of one project or library and contain a collection of Go packages that are then released together. Go modules **solve many problems with GOPATH , the original system, by allowing users to put their project code in their chosen directory and specify versions of dependencies for each module**.```

* make build


#  2 Primitive Types and Declarations
### Common Concepts Across BuiltIn Types
- zero value: a value assigned to declared but unassigned variable
    - strings --> empty string
    - boolean --> false
    -

- Literals
    -  writing out a number, character, or string

    - 4 Types
        1. Integer
            1. Usually based 10, but other bases are available:
                1. 0b for binary (base two), 0o for octal (base eight), or 0x for hexadecimal (base sixteen)
        2. Floating
            1. ex: 6.03e23

        3. Rune
            1. Characters that are surrounded by single quotes
                1. Single and double quoates are not interchangable
            2. Different types:
                1. single Unicode characters (`'a'`), 8-bit octal numbers (`'\141'`), 8-bit hexadecimal numbers (`'\x61'`), 16-bit hexadecimal numbers (`'\u0061'`), or 32-bit Unicode numbers (`'\U00000061'`)

        4. String

### Bullt Ins
- Go doesn’t allow automatic type promotion between variables because GO emphasizes clarity of intent and readability
``` GO
	var x int = 10
	var y float64 = 30.2
	var z float64 = float64(x) + y
	var d int = x + int(y)
```

### Declaring Variables
- Default
    - var x int = 10
- Since value can have defualt type, on can drop the type
    - var x = 10
- assign to zero values
    - var x, y int
- Declaration lists
```GO
	var (
	    x    int
	    y        = 20
	    z    int = 30
	    d, e     = 40, "hello"
	    f, g string
	)
```

- Short declaration
    - When you are ==insde a function==, one can use the := to replace a var assignment
```GO
		var x = 10
		x := 10
```

### Constants
- represents a value is immutable
- GO's peculariteis
    - Constants in Go are a way to give names to literals, which means they can be assigned to these values
```
	-   Numeric literals   
	-   `true` and `false`
	-   Strings
	-   Runes
	-   The built-in functions `complex`, `real`, `imag`, `len`, and `cap`
	-   Expressions that consist of operators and the preceding values
```

### Naming Variables
-  Like most languages, Go requires identifier names to start with a letter or underscore, and the name can contain numbers, underscores, and letters. Go’s definition of “letter” and “number” is a bit broader than many languages. Any Unicode character that is considered a letter or digit is allowed


#  3 Composite Types
### Array
	- Ex
```
	var x = [3]int{10,20,30}

	var x = [12]int{1, 5: 4, 6, 10: 100, 15}
	outut: [1, 0, 0, 0, 0, 4, 6, 0, 0, 0, 100, 15]
```


### Slice
* slice has no length, compared to arrays
* Good for sequential data
    - Ex:
```
		var x = []int{1, 5: 4, 6, 10: 100, 15}
		outut: [1, 0, 0, 0, 0, 4, 6, 0, 0, 0, 100, 15]
	
```

- append
  ```
      var x = []int{1, 2, 3}
      x = append(x, 4, 5)
  ```

- make
    -  create an empty slice that already has a length or capacity specified.
```
	x := make([]int, 5)
	x = append(x, 10)
```

### Strings
- Under the covers, Go uses a sequence of bytes to represent a strin
    - var s string = "Hello there"
    - var b byte = s[6]

### Maps
- Examples
```
	var nilMap map[string]int
	
	totalWins := map[string]int{}
	
	teams := map[string][]string {
	    "Orcas": []string{"Fred", "Ralph", "Bijou"},
	    "Lions": []string{"Sarah", "Peter", "Billie"},
	    "Kittens": []string{"Waldo", "Raul", "Ze"},
	}
	
	// I know there are 10 keys, but I don't know the values yet
	ages := make(map[int][]string, 10)

```
-  map returns the zero value if you ask for the value associated with a key that’s not in the map
```
	totalWins := map[string]int{}
	totalWins["Orcas"] = 1
	totalWins["Lions"] = 2
	fmt.Println(totalWins["Orcas"]) // prints 1
	fmt.Println(totalWins["Kittens"]) // prints 0, which is the zero value for ints
```

- but how do I know if 0 is because the value is truly zero, or bc key does not exist?
    - Comma ok idiom
```
		m := map[string]int{
		    "hello": 5,
		    "world": 0,
		}
		v, ok := m["hello"]
		fmt.Println(v, ok)
		
		v, ok = m["world"]
		fmt.Println(v, ok) // ok = true
		
		v, ok = m["goodbye"]
		fmt.Println(v, ok) // ok = false
```

### Structs

- Improves upon map by
    - constrain a map to only allow certain keys.
    - values in struct can be of different types
```
	type person struct {
	    name string
	    age  int
	    pet  string
	}

	var fred person

	julia := person{
	    "Julia",
	    40,
	    "cat",
	}

	beth := person{
	    age:  30,
	    name: "Beth",
	}
```


# 4 Blocks, Shadows, And Control Structures
### Blocks
- Go blocks allows variable declaration with scope
    - if inner scope collides/shadows an outer variable, the inner variable takes precedence
    - A shadowing variable is a variable that has the same name as a variable in a containing block. For as long as the shadowing variable exists, you cannot access a shadowed variable.
    - Ex
```
func main() {
    x := 10
    if x > 5 {
        fmt.Println(x)
        x := 5
        fmt.Println(x)
    }
    fmt.Println(x)
}
output: 10, 5, 10
```


# If
- very similar to other languages
```
	n := rand.Intn(10)
	if n == 0 {
	    fmt.Println("That's too low")
	} else if n > 5 {
	    fmt.Println("That's too big:", n)
	} else {
	    fmt.Println("That's a good number:", n)
	}
```
- But one can declare variables that are scoped to the condition and to both the `if` and `else` block
```
	if n := rand.Intn(10); n == 0 {
	    fmt.Println("That's too low")
	} else if n > 5 {
	    fmt.Println("That's too big:", n)
	} else {
	    fmt.Println("That's a good number:", n)
	}
```

### for loops  [iterations]
- There a 4 ways to iterate

- METHOD 1: for - range
    - returns an index and a value
```
	evenVals := []int{2, 4, 6, 8, 10, 12}
	for i, v := range evenVals {
	    fmt.Println(i, v)
	}
```

	  - iterate through a map
```
uniqueNames := map[string]bool{"Fred": true, "Raul": true, "Wilma": true}
for k, v= range uniqueNames {
    fmt.Println(k)
}

// if i only want the keys
for k: = range uniqueNames {
    fmt.Println(k)
}
```
	- the iterate variables are copies
```
evenVals := []int{2, 4, 6, 8, 10, 12}
for _, v := range evenVals {
    v *= 2
}
fmt.Println(evenVals) //[2 4 6 8 10 12]

```

### break and continue
- looks like the do while loop
```
	for {
	    // things to do in the loop
	    if !CONDITION {
	        break
	    }
	}
```

```
for i := 1; i <= 100; i++ {
    if i%3 == 0 && i%5 == 0 {
        fmt.Println("FizzBuzz")
        continue
    }
    if i%3 == 0 {
        fmt.Println("Fizz")
        continue
    }
    if i%5 == 0 {
        fmt.Println("Buzz")
        continue
    }
    fmt.Println(i)
}
```

- Condition only
```
i := 1
for i < 100 {
        fmt.Println(i)
        i = i * 2
}
```



### Switch
- when to use switch and if?
```
	A switch statement, even a blank switch, indicates that there is some relationship between the values or comparisons in each case.
```

```
words := []string{"hi", "salutations", "hello"}
for _, word := range words {
    switch wordLen := len(word); {
    case wordLen < 5:
        fmt.Println(word, "is a short word!")
    case wordLen > 10:
        fmt.Println(word, "is a long word!")
    default:
        fmt.Println(word, "is exactly the right length.")
    }
}
```

```python

# Find number of plants once can plant if
# 1 = there is a plant
# 0 = open spance
# One can only plan in non-adjacent splts
# Ex1: [1, 0, 0, 0, 1] --> cnt =1
# Ex2: [0, 0] --> cnt =1
def findNumPlants(flowerBeds):
	cnt = 0

	for i, v in enumerate(flowerBeds):
		if v == 0:
			left = max(i-1, 0)
			right = min(len(flowerBeds)-1, i+1)
			if flowerBeds[left]==1 or flowerBeds[right]==1:
				continue
			cnt += 1
			flowerBeds[i] = 1
	
	return cnt
	
- precsion vs recall vs AUC
- why is GPT more popular than Bert



def counterByNonLocal():
	c = 0
	candidates = []
	def count():
		nonlocal c
		nonlocal candidates
		candidates.append(11)
		c += 1
		return
	count()
	print(candidates, c)

counterByNonLocal() # [11], 1

def counterPassByReference():
	def count(cs, c):
		cs.append(11)
		c += 1
		return
	candidates = []
	index = 0
	count(candidates, index)
	print(candidates, index)	
counterPassByReference() # output: [11], 0
```
# 5 Functions
### Decalring and Calling Functions
- Optional parameters
    - GO does not have named and optional parameters
    - Example below achieves the same
``` go
		type MyFuncOpts struct {
		    FirstName string
		    LastName string
		    Age int
		}
		
		func MyFunc(opts MyFuncOpts) error {
		    // do something here
		}
		
		func main() {
		    MyFunc(MyFuncOpts {
		        LastName: "Patel",
		        Age: 50,
		    })
		}
```

- variadic input parameters and slices
    - enables one to pass n values of the same type
    - The variadic parameter must be the last (or only) parameter in the input parameter list. You indicate it with three dots (…) before the type.
``` go
	func addTo(base int, vals ...int) []int {
	    out := make([]int, 0, len(vals))
	    for _, v := range vals {
	        out = append(out, base+v)
	    }
	    return out
	}
```


### Functions are Values
- The type of a function is built out of the keyword func and the types of the parameters and return values. This combination is called the signature of the function. Any function that has the exact same number and types of parameters and return values meets the type signature.
- function type delcarations
    - Just like you can use the `type` keyword to define a `struct`, you can use it to define a function type, too
``` go
var opMapV1 = map[string]func(int, int) int{
    "+": add,
    "-": sub,
    "*": mul,
    "/": div,
}

type opFuncType func(int,int) int
var opMapV2 = map[string]opFuncType {
	...
}
```

### Closures
- Closure are function defined in another function
    - The inner functions have access and can modify variables declared in the outer function
``` go
		type Person struct {
		    FirstName string
		    LastName  string
		    Age       int
		}
		
		people := []Person{
		    {"Pat", "Patterson", 37},
		    {"Tracy", "Bobbert", 23},
		    {"Fred", "Fredson", 18},
		}
		
		// sort by age
		// outer function is sort.Slce; inner function is func(i int, j int)
		sort.Slice(people, func(i int, j int) bool {
		    return people[i].Age < people[j].Age
		})
```



### Defer
- In Go, the cleanup code is attached to the function with the `defer` keyword
``` go
func main() {
    if len(os.Args) < 2 {
        log.Fatal("no file specified")
    }
    f, err := os.Open(os.Args[1])
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    data := make([]byte, 2048)
    for {
        count, err := f.Read(data)
        os.Stdout.Write(data[:count])
        if err != nil {
            if err != io.EOF {
                log.Fatal(err)
            }
            break
        }
    }
}
```

### Go is Call by Value
- When you supply a variable for a parameter to a function, Go always makes a copy of the value of the variable
``` go
type person struct {
    age  int
    name string
}

func modifyFails(i int, s string, p person) {
    i = i * 2
    s = "Goodbye"
    p.name = "Bob"
}

func main() {
    p := person{}
    i := 2
    s := "Hello"
    modifyFails(i, s, p)
    fmt.Println(i, s, p)
}

Output: 2 Hello {0 }

```


# 6 Pointers
- A pointer is simply a variable that holds the location in memory where a value is stored
- Example
``` go
	var x int32 = 10
	var y bool = true
	pointerX := &x
	pointerY := &y
	var pointerZ *string
```

- ==pointers indicate mutable parameters==
    - go is a call by value, which gives rise to immutability
      ```
      Immutable types are safer from bugs, easier to understand, and more ready for change. Mutability makes it harder to understand what your program is doing, and much harder to enforce contracts.
  
      Mutable objects is just fine if you are using them entirely locally within a method, and with only one reference to the object
      ```
        - EXAMPLE
``` go
		func failedUpdate(px *int) {
		    x2 := 20
		    px = &x2
		}
		
		func update(px *int) {
		    *px = 20
		}
		
		func main() {
		    x := 10
		    failedUpdate(&x)
		    fmt.Println(x) // prints 10
		    update(&x)
		    fmt.Println(x) // prints 20
		}
```



# 7 Types, Methods, and Interfaces
- GO avoids inhertiance by encouraging compositions.  This is accomplished via types, methods, and interfaces.
    - Like most languages, Go allows you to attach methods to types. It also has type abstraction, allowing you to write code that invokes methods without explicitly specifying the implementation
- User Defined Types
    - IMOW:
        - lessen the use of classes
        - makes code more readable

    - Examples
``` go
		type Person struct {
		    FirstName string
		    LastName string
		    Age int
		}
		
		type Score int
		type Converter func(string)Score  // Interesting
		type TeamScores map[string]Score  
```

- Use the user define types in functions
```go
	func (p Person) String() string {
	    return fmt.Sprintf("%s %s, age %d", p.FirstName, p.LastName, p.Age)
	}
```

### Method vs Functions
- methods are associated with user define types
    - func (RECEIVER) methodName( params ...) returnType { logic }
    - RECEIVER refers to the user defined type.  Functions, in comparison, has no RECEIVER.
``` go
	type Person struct {
	    FirstName string
	    LastName string
	    Age int
	}
	
	func (p Person) String() string {
	    return fmt.Sprintf("%s %s, age %d", p.FirstName, p.LastName, p.Age)
	}

	output := p.String() //calling the method

```

- method and function declration cannot be overloaded.

- When to use pointer in the receiver?
    - Factors
```
		- If your method modifies the receiver, you must use a pointer receiver.
		
		- If your method needs to handle nil instances (see “Code Your Methods for nil Instances”), then it must use a pointer receiver.
		
		- If your method doesn’t modify the receiver, you can use a value receiver.
```

``` go
	type Counter struct {
	    total             int
	    lastUpdated time.Time
	}
	
	func (c *Counter) Increment() {
	    c.total++
	    c.lastUpdated = time.Now()
	}
	
	func (c Counter) String() string {
	    return fmt.Sprintf("total: %d, last updated: %v", c.total, c.lastUpdated)
	}
	
	var c Counter
	fmt.Println(c.String())
	c.Increment()
	fmt.Println(c.String())
	
	total: 0, last updated: 0001-01-01 00:00:00 +0000 UTC
	total: 1, last updated: 2009-11-10 23:00:00 +0000 UTC m=+0.000000001

```
- Relationship between method and functions?
    - A function can be "aliased" to a method
``` go
		type Adder struct {
		    start int
		}
		
		func (a Adder) AddTo(val int) int {
		    return a.start + val
		}
		
		myAdder := Adder{start: 10}
		fmt.Println(myAdder.AddTo(5)) // prints 15
		
		f1 := myAdder.AddTo
		fmt.Println(f1(10))           // prints 20
```

- When should one use function vs method?
    - Prefer function when logic depends only on input parameter

- Go enumeration is achieved with ==iota==
``` go
	type MailCategory int
	
	const (
		Uncategorized MailCategory = iota
		Personal
		Spam
		Social
		Advertisements
	)
```

### Use Embeddings for Composition
- Acheive composition via embedded field.
- An embedded field is a field in a struct which has no name
 ``` go
	type Employee struct {
		Name         string
		ID           string
	}
	
	func (e Employee) Description() string {
		return fmt.Sprintf("%s (%s)", e.Name, e.ID)
	}
	
	type Manager struct {
		Employee // <-- embedded field; it's only a type !
		Reports []Employee
	}
	
	m := Manager{
		Employee: Employee{
			Name:         "Bob Bobson",
			ID:             "12345",
		},
		Reports: []Employee{},
	}
	fmt.Println(m.ID)            // prints 12345
	fmt.Println(m.Description()) // prints Bob Bobson (12345)
```
- ==Embeddings is NOT inheritance==. You cannot assign a variable of type `Manager` to a variable of type `Employee`

### Interfaces
- GO interface are imiplicity
    - Enable both type-safety and decoupling, bridging the functionality in both static and dynamic languages
    -  In GO, a type implements the interface IF it has all the method set defined in an interface.
        - IMOW: ==GO interface sounds like duck typing for static languages.==
- A struct implements an interface if every method in struct
    - function name is same
    - same number of input parameters
    - each input parameter. has the same type
    - but the input parameter name can be different
```go
	type Somether interface {
		Method(x string) bool
	}
	
	type MyType string
	func (mt MyType) Method(y string) bool {
		return true
	}
	
	func main() {
		val := MyType("hello")
		var i interface{} = val
	
		v, ok := i.(Somether) // chech if val is of the type Somether interface. We can also use a interface case switch statement below
		fmt.Println(v, ",", ok) //ok = true
	}

```

- [how to check value implements interface](https://stackoverflow.com/questions/27803654/explanation-of-checking-if-value-implements-interface)
  - 
- Dynamic langauges do not have interfaces; they use duck typing
    - The concept is that you can pass an instance of a type as a parameter to a function as long as the function can find a method to invoke that it expects:
- Turns out both the static and dynamic camps have valid points
    - static (java): interface makes code more readable bc the interface defines all the expected functionality
    - dynamic camp (python): it's easier to refactor the implementations
    - `If your application is going to grow and change over time, you need flexibility to change implementation. However, in order for people to understand what your code is doing (as new people work on the same code over time), you also need to specify what the code depends on.

- Ex1:
``` go
	type LogicProvider struct {}
	
	func (lp LogicProvider) Process(data string) string {
	    // business logic
	}
	
	type Logic interface {
	    Process(data string) string
	}
	
	type Client struct{
	    L Logic
	}
	
	func(c Client) Program() {
	    // get data from somewhere
	    c.L.Process(data)
	}
	
	main() {
	    c := Client{
	        L: LogicProvider{},
	    }
	    c.Program()
	}
```
`
- Ex2: interace can embed other interfaces
```go
	type Reader interface {
	        Read(p []byte) (n int, err error)
	}
	
	type Closer interface {
	        Close() error
	}
	
	type ReadCloser interface {
	        Reader
	        Closer
	}
```

- Best practices
    - Code conventions: interface names end in ER
    - ==Code should accept interfaces and return structs==

### Type assertion and Type Switches
- 2 ways to check if a variable is of an interface type
- Method 1: Via OK idiom
```go
	type MyInt int

	var i interface{}
    var mine MyInt = 20
    i = mine

	i2, ok : = i.(int)
	if !ok { ... }
```

- Method 2: type switch
```go
	func doThings(i interface{}) {
	    switch j := i.(type) {
	    case nil:    // i is nil, type of j is interface{}
	    case int:    // j is of type int
	    case MyInt:  // j is of type MyInt
	    case string: // j is a string
	    case bool, rune: // i is either a bool or rune, so j is of type interface{}
	    default: // no idea what i is, so j is of type interface{}
	    }
	}
```


### Dependency Injection Via Implicit Interfaces
- dependency injection pattern helps decouple
    - functionality of objects
    - constructor of those objects
    - to help application change with time
```go
// -----------------------------------
// Driver code
// -----------------------------------
func main() {
    l := LoggerAdapter(LogOutput)
    ds := NewSimpleDataStore()
    logic := NewSimpleLogic(l, ds) //New* are factory methods
    c := NewController(l, logic)
    http.HandleFunc("/hello", c.SayHello)
    http.ListenAndServe(":8080", nil)
}

// -----------------------------------
// Log related
// -----------------------------------
func LogOutput(message string) {
    fmt.Println(message)
}

type Logger interface {
    Log(message string)
}

// adapter adapts the LogOutput function to the Logger interface
type LoggerAdapter func(message string)
func (lg LoggerAdapter) Log(message string) {
	//NOTE: LoggerAdapter implements the Logger interface bc it contains all the methods in the Logger interface; method's input parameter has the same type
    lg(message) // calls the func in type LoggerAdapter
}

// ------------------------------
// Datastore
// ------------------------------
type SimpleDataStore struct {
    userData map[string]string
}
func (sds SimpleDataStore) UserNameForID(userID string) (string, bool) {
    name, ok := sds.userData[userID]
    return name, ok
}

// Factory method
func NewSimpleDataStore() SimpleDataStore {
    return SimpleDataStore{
        userData: map[string]string{
            "1": "Fred",
            "2": "Mary",
            "3": "Pat",
        },
    }
}

// interface
type DataStore interface {
    UserNameForID(userID string) (string, bool)
}

// ------------------------------
// LOGIC
// ------------------------------
type SimpleLogic struct {
    l  Logger
    ds DataStore
}
func (sl SimpleLogic) SayHello(userID string) (string, error) {
    sl.l.Log("in SayHello for " + userID)
    name, ok := sl.ds.UserNameForID(userID)
    if !ok {
        return "", errors.New("unknown user")
    }
    return "Hello, " + name, nil
}
func (sl SimpleLogic) SayGoodbye(userID string) (string, error) {
    sl.l.Log("in SayGoodbye for " + userID)
    name, ok := sl.ds.UserNameForID(userID)
    if !ok {
        return "", errors.New("unknown user")
    }
    return "Goodbye, " + name, nil
}

//factory method for logic
func NewSimpleLogic(l Logger, ds DataStore) SimpleLogic {
    return SimpleLogic{
        l:    l,
        ds: ds,
    }
}

//interface
type Logic interface {
    SayHello(userID string) (string, error)
}


// ------------------------------
// Controller
// ------------------------------
type Controller struct {
    l     Logger
    logic Logic
}
func (c Controller) SayHello(w http.ResponseWriter, r *http.Request) {
    c.l.Log("In SayHello")
    userID := r.URL.Query().Get("user_id")
    message, err := c.logic.SayHello(userID)
    if err != nil {
        w.WriteHeader(http.StatusBadRequest)
        w.Write([]byte(err.Error()))
        return
    }
    w.Write([]byte(message))
}

//Interface
func NewController(l Logger, logic Logic) Controller {
    return Controller{
        l:     l,
        logic: logic,
    }
}
```




# 8 Errors
- GO does not have exception handling
    - Go handles errors by returning a value of type error as the last return value for a function
    - when a function executes as expected, nil is returned for the error parameter. If something goes wrong, an error value is returned instead
```go
func calcRemainderAndMod(numerator, denominator int) (int, int, error) {
    if denominator == 0 {
        return 0, 0, errors.New("denominator is 0")
    }
    return numerator / denominator, numerator % denominator, nil
}

func main() {
    numerator := 20
    denominator := 3
    remainder, mod, err := calcRemainderAndMod(numerator, denominator)
    if err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
    fmt.Println(remainder, mod)
}
```

- error is an interface
```go
type error interface {
    Error() string
}
```

### Sentinel Errors

### Error are Values

### Wrapping Errors


### Panic -> Defer -> Recover
- Go generates a panic whenever there is a situation where the Go runtime is unable to figure out what should happen next
```go
func DoSomeThings(val1 int, val2 string) (_ string, err error) {
    defer func() {
        if err != nil {
            err = fmt.Errorf("in DoSomeThings: %w", err)
        }
    }()
    val3, err := doThing1(val1)
    if err != nil {
        return "", err
    }
    val4, err := doThing2(val2)
    if err != nil {
        return "", err
    }
    return doThing3(val3, val4)
}
```

- When panic happens, the current function exits immediately and any defers attached to the current function start running.
    - panic generates the stack trace
```go
	func doPanic(msg string) {
	    panic(msg)
	}
	
	func main() {
	    doPanic(os.Args[0])
	}
	
	panic: /tmpfs/play
	
	goroutine 1 [running]:
	main.doPanic(...)
	    /tmp/sandbox567884271/prog.go:6
	main.main()
	    /tmp/sandbox567884271/prog.go:10 +0x5f
```
- recover can be called from the defer function
```go
	func div60(i int) {
	    defer func() {
	        if v := recover(); v != nil {
	            fmt.Println(v)
	        }
	    }()
	    fmt.Println(60 / i)
	}
	
	func main() {
	    for _, val := range []int{1, 2, 0, 6} {
	        div60(val)
	    }
	}
```

# 9 Modules, Packages, and Imports
### Repositories, Modules and Packages
- repository:  a version control system where source code for a project is stored
- Module: the root of a Go library or application stored in a repository
    - ==modules = P packages==
- Packages have paths, so we can use code outside of the module
    - path: use to access the module;
        - GO: github.com/jonbodner/proteus
        - JAVA: com.companyname.projectname.library


### go.mod
- Source code becomes a module when it has a go.mod.
    - `cmd: `go mod init` _`MODULE_PATH`_`
    - looks like a setup.py in python
``` text
	module github.com/learning-go-book/money
	
	go 1.15
	
	require (
	    github.com/learning-go-book/formatter v0.0.0-20200921021027-5abc380940ae
	    github.com/shopspring/decimal v1.2.0
	)
```



### Building Packages
- Imports and Exports
    - `import` statement allows you to access exported constants, variables, functions, and types in another package
    - `import`  [packages]
        - Variables with _capitalization_ means it's package-level identifier AND  is visible outside of the package
``` go
	// 1st package print has this package path:
	// "github.com/learning-go-book/package_example/formatter"
	package print
	import "fmt"
	func Format(num int) string {
	    return fmt.Sprintf("The number is %d", num)
	}
	
	
	// 2nd package named main
	package main
	import (
	    "fmt"
	    "github.com/learning-go-book/package_example/formatter"
	)
	
	func main() {
	    output := print.Format(num)
	    fmt.Println(output)
	}
```
- How to organize your module
    - When your module is small, keep all of your code in a single package.
    - If your module consists of one or more applications, create a directory called cmd at the root of your module. Within cmd, create one directory for each binary built from your module
    - If your module’s root directory contains many files for managing the testing and deployment of your project (such as shell scripts, continuous integration configuration files, or Dockerfiles), place all of your Go code (besides the main packages under cmd) into packages under a directory called pkg
        - Within the pkg directory, organize your code to limit the dependencies between packages. One common pattern is to organize your code by slices of functionality. For example, if you wrote a shopping site in Go, you might place all of the code to support customer management in one package and all of the code to manage inventory in another.
        - [youtube](https://www.youtube.com/watch?v=oL6JBUk6tj0)

- Override package name, when packages collide
```go
	import (
	    crand "crypto/rand" // refer crypto/rand as crand
	    "encoding/binary"
	    "fmt"
	    "math/rand"
	)
	
	func seedRand() *rand.Rand {
	    var b [8]byte
	    _, err := crand.Read(b[:])
	    if err != nil {
	        panic("cannot seed with cryptographic random number generator")
	    }
	    r := rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(b[:]))))
	    return r
	}
```

- Documentation
    - tools
        - `golint` and `golangci-lint` can report missing comments on exported identifiers.
        - go doc
            - The command go doc PACKAGE_NAME displays the package godocs for the specified package and a list of the identifiers in the package. Use go doc PACKAGE_NAME.IDENTIFIER_NAME to display the documentation for a specific identifier in the package.

- ==How to share code between packages?  Ans: internal packages==
    - Identifiers in an internal package are visible to that package and its subpackages are only accessible to the direct parent package of `internal` and the sibling packages of `internal`
    - [github example](https://github.com/learning-go-book/internal_example/blob/master/bar/bar.go)
    -
``` 
	/bar/bar.go
	
	/foo/
		internal/
			internal.go
			deeper/deper.go //can acces internal.go
				
		sibling/
			sibling.go //can access internal.go
```

- circular dependencies
    -  In some cases, this is caused by splitting packages up too finely. If two packages depend on each other, there’s a good chance they should be merged into a single package. We can merge our person and pet packages into a single package and that solves our problem.
    - If you have a good reason to keep your packages separated, it may be possible to move just the items that cause the circular dependency to one of the two packages or to a new package

### Working with Modules
- Import 3rd party modules
    - go.mod requires the 3rd party module
```go
		module region_tax
		
		go 1.15
		
		require (
		    github.com/learning-go-book/simpletax v1.1.0
		    github.com/shopspring/decimal v1.2.0
		)
```
			- or via terminal: $ go get github.com/learning-go-book/simpletax@v1.0.0
				- this will update the go.mod

- how choose the package version
    - module system uses the principle of _minimal version selection_

- manually controlling which version one wants
    - intial:
        - go get github.com/learning-go-book/simpletax@v1.1.0
    - upgrade/patch:
        - go get -u=patch github.com/learning-go-book/simpletax
    - get the latest version
        - go get -u github.com/learning-go-book/simpletax
- [GO module wiki](https://github.com/golang/go/wiki/Modules)

### Publishing Your Module

### Versioning Your Module

### Module Proxy Server


### Summary



# 10 Concurrency
- Concurrency in go is mainly attained through channels and select
    - go also has mutex and atomic
- Philosophy:
    - Share memory by communicating; do not communicate by sharing memory.
        - mutex and atomic shares memory


# 11 The Standard Library



# 12 The Context



# 13 Writing Tests


# 14 Reflect, Unsafe, and

# 15 Appendix
## 15.1  References
- Go + Torch + Triton
    - [github](https://github.com/miguelvr/trtserver-go)
- [Akka vs Go concurrency](https://gist.github.com/scottfrazer/094a0f1d048cc3b8deeac4cf29266f62)
- Cross reference to [[Tool - Python - GO]]

## 15.2 Dependencies and modules
- [ref](https://www.digitalocean.com/community/tutorials/how-to-use-go-modules)
- Terminology
    - Module:
        - IMOW: a library we create.  Damian's CLI is a module

    - Package
        - subdirectories inside a module
- Go files can refer to local  or remote modules
  ``` Go
  package main

  import (
      "fmt"
      // Remote module; we can specify see below for different variabts: git tag, branch nrame, etc..
      "github.com/spf13/cobra"

      // Local current module
      "mymodule/mypackage"
  )
  
  func main() {
      cmd := &cobra.Command{
          Run: func(cmd *cobra.Command, args []string) {
              fmt.Println("Hello, Modules!")
  
              mypackage.PrintHello()
          },
      }
  
      fmt.Println("Calling cmd.Execute()!")
      cmd.Execute()
  }
  ```
    - Before you can  import the remote module, one need to "go get"
        - So which version does one get?
            - default: get latest
                - _go get github.com/spf13/cobra_
            - a specific commit
                - _go get github.com/spf13/cobra@latest_
                - _go get github.com/spf13/cobra@lfeature_branch
                - go get github.com/spf13/cobra@l[GIT-COMMIT-ID]
                - go get github.com/spf13/cobra@l[GIT-TAG-ID]
- When you run a go file, it updates the go.sum with records the hash and version of the dependencies used in the run
- Flow: Summar

  ```mermaid
  graph LR;
      module-remote -- cmd: go get .. --> go.mod;
      file.go -- cmd: go run file.go --> go.sum		 
  ```
	
