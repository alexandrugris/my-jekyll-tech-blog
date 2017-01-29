---
layout: post
title:  "C++ Play - my own short guide to C++11/14 (Part 2)"
date:   2017-01-29 12:15:16 +0200
categories: Native coding
---

In the previous blog post I covered a lot of ground: variadic templates, improvements to classes, improvements to standard library, smart pointers and lambdas. 
We also briefly touched move semantics as well as decltype and declval. In this part I am going to continue with move, delegated constructors, user defined literals and static_assert. In the next article 
I will write about local storage for threads while briefly touching thread support in the new standard library.

### Move

The addition of move semantics was a major change to the language which had rippling effects on how libraries are designed, how code is written and how it performs. 
While it adds a layer of complexity to the language and how surely lenghens the code of libraries, the usage becomes much more straight forward and clear in intent.

Move solves several problems:

- Storing a full type in a std:vector was in most of the cases a no-go, due to the overhead associated with repeated construction and destruction of each object, whenever the vector increased in size. 
Therefore we usually stored only pointers (or smart pointers :) ) in such containers. 
- In order to have move-like semantics, one had o open the implementation and re-target the copy operators to move pointers or, in best case, add copy-on-write. This either made the usage of the objects non-obvious and non-uniform (the copy is not really a copy) or added another layer of complexity for the copy-on-write implementation.
- Libraries (for instance maths) had to be coded in convoluted ways to avoid unnnecessary return copies. For instance, one might simply avoid overloading the `operator+` for a matrix class and only provide the `operator +=` just to make sure no unnecessary allocations are performed when not needed. This might sound like excessive optimization, but, hey, C++ was designed with the idea of maximum performance everywhere, so why squander it when it can be avoided? 

The addition of move brought in a another nice side benefit. From a esthetical point of view, I personally like the expressiveness of [fluent-style](https://en.wikipedia.org/wiki/Fluent_interface) coding. 
Move makes it easy to have high-performance code written like this.

But let's start with the beginning:

*Move syntax*

```csharp
struct MyMoveStuff {

	MyMoveStuff() {
		cout << "  Constructor" << endl;
	}

	~MyMoveStuff() noexcept {
		cout << "  Destructor" << endl;
	}

	MyMoveStuff(const MyMoveStuff&) {
		cout << "  Copy constructor" << endl;
	}

	// noexcept guarantees to the compiler that all exceptions are treated internally, 
	// without thowing anything. 
	// Needed for move operations. 
	// There is also the operator noexcept which returns true 
	// if the operations inside do not throw exceptions. 
	// e.g. 
	// template<typename t> 
	// auto square(const T& t) noexcept(noexcept(t * t)) -> decltype(t*t) { return t * t; }	
	MyMoveStuff(MyMoveStuff&& t) noexcept { 
		cout << "  Move constructor" << endl;

		// ATTENTION!!!!! Great source of bugs if the case is not handled properly !!!!
		if (this == &t) { 
			cout << "Moving into itself -> leave intact" << endl;
		}
	}	

	static MyMoveStuff create_through_variable() {
		MyMoveStuff m;
		return m;
	}

	static MyMoveStuff create_no_variable() {
		return MyMoveStuff();
	}
};
```

To keep things simple, we are sticking only with the constructors for the demo, 
but please consider implemeting also `operator=(MyMoveStuff&& move) noexcept`. 
Just like you would do with `operator=(const MyMoveStuff& m)`.

Now let's run this test:

```csharp
#define TEST(code) cout << #code << ": " << endl; code; cout << endl;

void test_move() {

	// test with vector
	cout << "Test with vector" << endl << "=========================" << endl;
	{
		TEST(MyMoveStuff m);

		TEST(std::vector<MyMoveStuff> vec = { MyMoveStuff() });

		// reserve does not use move semantics (uses copy constructor) inside 
		// UNLESS move constructor and destructor are marked with noexcept 
		TEST(vec.reserve(100));				

		TEST(vec.push_back(m));		// copy constructor
		
		TEST(vec.emplace_back(m));	// copy constructor

		TEST(vec.emplace_back());	// constructor

		TEST(vec.emplace_back(MyMoveStuff())); // constructor, move, destructor

		TEST(vec.push_back(move(m)));// move constructor

		TEST(vec.push_back(move(MyMoveStuff()))); // constructor, move , destructor

		TEST(vec.push_back(MyMoveStuff())); // constructor, move, destructor

		TEST(vec.clear());					
	}

	// if reserve is not used, billions of allocations, deallocations occur
	// if list<> is used instead of vector<> in the code above, 
	// its execution is identical to std::vector followed by vec.reserve

	//////////

	cout << endl << "Test return from functions" << endl << "=========================" << endl;
	{
		// just the constructor
		TEST(MyMoveStuff m = MyMoveStuff::create_no_variable());

		// constructor followed by move
		TEST(MyMoveStuff m2 = MyMoveStuff::create_through_variable()); 
	}

	// test with inheritance
	class MyMoveStuff2 : public MyMoveStuff {
	public:

		MyMoveStuff2() : MyMoveStuff() {}

		// transformation is needed; otherwise copy constructor will be called
		// std::move is implemented like this:
		// template<typename T> constexpr T&& move(T&& t) { return t; }
		MyMoveStuff2(MyMoveStuff2&& value) : MyMoveStuff(move(value)) { 

		}
	};

	cout << endl << "Test with inheritance" << 
		endl << "=========================" << endl;
	{
		TEST(MyMoveStuff2 m);
		TEST(MyMoveStuff2 m2(move(m)));
	}
}
```

*Fluent-style*

Fluent style is possible due to the introduction of reference qualifiers for member functions. That is, the `&` and `&&` operators appended to the end of the function declaration.
C++ makes a clear distiction between lvalues and rvalues ([discussion here, from MSDN](https://msdn.microsoft.com/en-us/library/f90831hc.aspx)). In short: 

> Every C++ expression is either an lvalue or an rvalue. An lvalue refers to an object that persists beyond a single expression. 
> You can think of an lvalue as an object that has a name. All variables, including nonmodifiable (const) variables, are lvalues. 
> An rvalue is a temporary value that does not persist beyond the expression that uses it. 


 ```csharp
void test_reference_qualifiers_for_member_functions() {

	class WithFluentProgrammingStyle : public MyMoveStuff{
	public:

		using MyMoveStuff::MyMoveStuff;

		// on lvalue, do not change the initial object
		WithFluentProgrammingStyle operation() const & {	
			return *this;
		}

		// on rvalue, allow the object to pass through, 
		// withount needs for copy intermediate objects to be created and destructed
		WithFluentProgrammingStyle operation() && {			 
			return move(*this);
		}

		WithFluentProgrammingStyle 
		operator + (const WithFluentProgrammingStyle& other) const & {
			return *this; // Copy
		}

		WithFluentProgrammingStyle 
		operator +(const WithFluentProgrammingStyle& other) && {
			return move(*this); // move
		}

	};

	TEST(WithFluentProgrammingStyle fluent);

	TEST(fluent.operation().
				operation().
				operation().
				operation());

	TEST(WithFluentProgrammingStyle fluent2 = fluent + fluent + fluent);
}
 ```

 ### Inheriting constructors, delegated constructors, static_assert, exceptions

*Inheriting constructors*

```csharp
class Derived : Base  
{  
public:  
    // Inherit all constructors from Base  
    using Base::Base;    
private:      
    int private_member = 0;  // x is initialized inline
};  
```

*Delegated constructors*

Please note the private `Test2(int x, int y)` and the invocation `Test2() : Test2(0, 0)`

```csharp
class Test2 { // Delegated constructors
public:
	Test2() : Test2(0, 0){
		cout << "Test2 default constructor" << endl;
	}
	Test2(Test2& tst2) : Test2(1, 1){
		cout << "Test2 copy constructor" << endl;
	}

private:
	Test2(int x, int y){
		cout << "Test2 delegated constructor" << endl;
	}
};
```

*Static asserts*

This is a topic which is worth going into deeper details, but I am just going to hint at it through a very simple example. The same example also hints at type-traits.

```csharp
class AX {
public:
	virtual ~AX() {}
};

class B : public AX {
public:
	B() {
		// TODO: check type traits
		// if the AX class does not have a virtual destructor, the code below 
		// generates a compile error
		static_assert(has_virtual_destructor<AX>::value, 
			"A must have a virtual destructor"); 
	}
};
```

*Exceptions - not new, but worth mentioning anyway*

```csharp
void test_exceptions() {
	try {

		auto pTest = make_shared<Test>(1, 2, "Trei"); // RAII
		throw exception("Hello World! :)");

	}
	// catch as const &; if not caught as "&", you suffer slicing, 
	// that is you lose part of the class that was derived from exception. 
	// you only get the base class.
	catch (const std::exception& ex) { 
		cout << ex.what() << endl;
	}
}
```

Notable here is the `noexcept` keyword which marks that the method does not throw an exception. See above for use, in the "move" section. From the 
[CPP Reference](http://en.cppreference.com/w/cpp/language/noexcept_spec):

> The compiler can use this information to enable certain optimizations on non-throwing functions as well as enable the noexcept 
> operator, which can check at compile time if a particular expression is declared to throw any exceptions. 
> For example, containers such as `std::vector` will move their elements if the elements' move constructor 
> is noexcept, and copy otherwise (unless the copy constructor is not accessible, but a potentially 
> throwing move constructor is, in which case the strong exception guarantee is waived). 

### User defined literals

This is especially useful for code which relies on units conversion. For instance physics code. It allows type-safe conversions and compile-type error checking. Here is an example, just to wet your appetite:

```csharp
class Km;

class M {
	double value;
public:
	constexpr M(const double& d) : value(d) {}
	operator double() { return value; }
	operator Km();
};

class Km {
	double value;
public:
	constexpr Km(const double& d) : value(d) {}
	operator double() { return value; }
	operator M() { return M(value * 1000.0f); }
};

M::operator Km() {
	return value / 1000.0f;
}

constexpr Km operator "" _km(long double d) {
	return d;
}
constexpr M operator "" _m(long double d) {
	return d;
}

void test_user_defined_literals() {

	Km dist_in_km = 10.5_km;	

	M dist_in_m = dist_in_km;
	dist_in_km = dist_in_m;

	cout << "The distance is " << dist_in_m << "m" << endl;
	cout << "The distance is " << dist_in_km << "km" << endl;
}
```