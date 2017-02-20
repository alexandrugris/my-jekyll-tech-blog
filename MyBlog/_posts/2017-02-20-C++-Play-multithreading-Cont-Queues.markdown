---
layout: post
title:  "C++ Play - Multithreading (Queues)"
date:   2017-02-20 12:15:16 +0200
categories: Native coding
---
In this post I am going to build a multithreaded queue to exemplify various issues regarding synchronization. 
I am going to measure the cost of locking and thread contention and provide some ideas to improve performance. 
I am going to build the tests incrementally, refining the solution as I progress through the code, 
but I will also skip some steps to keep the post meaningfully short.

### First, testing

In order to test the correctness of the application, I will use two types:

- A test type, defined by me, with enough weight inside to allow performance tests to have a baseline to run against (compare the time spent in locking to actual work of copying and moving the object in and out of the queue). I will name this type `TestType`. :)
- A `unique_ptr<TestType>` to validate against the common scenario of having a pointer stored in the queue, as well as making sure the move semantics work properly. 

For the `TestType` I am going to keep the type as simple as possible, with only move enabled, just to make sure all unnecessary copies are kept under control - deep copy errors will be found at compile time. Here is the definition:

```csharp
class TestType {
public:

	int arr[50]; // put some weight into the object

	TestType() {		
		iota(arr, arr + sizeof(arr) / sizeof(int), 0);
	}

	~TestType() {}

	TestType(const TestType& t) = delete;

	TestType(TestType&& t) {
		memcpy(arr, t.arr, sizeof(arr));
	}

	TestType& operator =(TestType&& t) {
		memcpy(arr, t.arr, sizeof(arr));
		return *this;
	}

public:
	void not_null() {
		cout << "Yeey, not null?: " << hex << this << endl;
	}
};
```

### The most basic queue:

A simple multithreaded queue with all operations synchronized with a mutex. I went for some design choices from the beginning:

- I wanted be able to use various synchronization primitives as long as they respect the `std::mutex` and `std::condition_variable` interface. I wanted to be to play around and alter the default behavior.
- I wanted to be able to track memory allocations. My initial assumption was that optimizing memory allocations will play a significant part in the process of streamlining the queue. Therefore I built my own allocator, which simply wraps `malloc` and `free`. More on this later.
- I wanted a very straight-forward interface, encapsulating all the behavior. 

```csharp
template<typename T, typename mtx_type, typename cond_variable_type> class blocking_queue {
private:
	std::deque<T, my_allocator<T>> _deque;
	mtx_type mtx;
	cond_variable_type cv;

public:

	blocking_queue() { }
	blocking_queue(blocking_queue&& other) : _deque(move(other._deque)) { }
	blocking_queue(const blocking_queue& other) = delete;

	T get() {
		std::unique_lock<mtx_type> lock(mtx);
		while (_deque.size() == 0)
			cv.wait(lock);

		T t = move(_deque.front()); _deque.pop_front();
		return move(t);

	}

	void put(T&& t) {
		std::unique_lock<mtx_type> lock(mtx);
		_deque.push_back(move(t));
		cv.notify_one();
	}
	void put(const T& t) {
		std::unique_lock<mtx_type> lock(mtx);
		_deque.push_back(t);
		cv.notify_one();
	}
  
  	bool try_get(char uninitialized[sizeof(T)]) {
		std::unique_lock<mtx_type> lock(mtx);

		if (_deque.size() > 0) {
			new (uninitialized) T(move(_deque.front()));
			_deque.pop_front();
			return true;
		}

		return false;
	}

	bool try_get(T& out) {
		std::unique_lock<mtx_type> lock(mtx);

		if (_deque.size() > 0) {
			out = move(_deque.front());
			_deque.pop_front();
			return true;
		}

		return false;
	}
  
};
```

Design considerations:

- I specifically omited a method `size()` to return the numer of elements in the queue. Being in a multithreaded context, this method would not make much sense.
- Replacement for the method `size()` is the `try_get()` which will lock the queue and atomically return an element if one exists. This design will have further implications down the stream, when I will use the queue in a multi-queue configuration and I will be forced to use external locking to obtain the minimum contention. The other option would have been an implementation that follows the pattern `try_lock` then `if(size() > 0) return element;` but in this case there would have been no hint on the actual size of the queue. So being in doubt on the best design, I kept locking external in the multi-queue and preserved the method as it is. 
- I added two `try_get`s. One of them accepts a stack allocated block of memory as input parameter to hint to the caller to avoid default-constructing the receiving object of type `T`. The method will invoke the move constructor itself on that memory. 
