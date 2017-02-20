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

### Testing

1. Very basic tests, just to make sure data isn't lost on transfer between the callers and the queue.

```csharp
void test_basic_bq() {

	typedef blocking_queue<unique_ptr<TestType>, std::mutex, std::condition_variable> MutexBlockingQueue;

	{
		MutexBlockingQueue bq;
		auto p = make_unique<TestType>();
		p->not_null();

		bq.put(move(p));
		p->not_null();

		p = bq.get();
		p->not_null();
	}

	{
		MutexBlockingQueue b;
		b.put(make_unique<TestType>());
		MutexBlockingQueue bq = move(b);
	}
}
```

2. Multithreading tests:

Aim is to combine in a simple manner various parameters. Here is an example of a possible `main` function:

```csharp
int main()
{

	test_basic_bq();

	test_mt_bq<blocking_queue, TestType>(0);
	test_mt_bq<blocking_queue, unique_ptr<TestType>>(0);
	test_mt_bq<blocking_queue, TestType, std::mutex, std::condition_variable>(0);

	test_mt_bq<multi_blocking_queue, TestType>();

	return 0;
}
```

I am instatiating the `test_mt_bq` with various parameters:

1. Simple blocking queue (the one I spoke about before) with `TestType` - default it uses windows critical sections as synchronization mechanism (see below)
2. Simple blocking queue with `unique_ptr<TestType>`
3. Simple blocking queue with `std::mutex` and `std::condition_variable`
4. Multi-queue blocking queue with default synchronization (windows critical sections) and `TestType`

Below is the test function:

```csharp
template<template<class...> class QueueType, 
			typename TT, 
			typename mtx_type = critical_section_win, 
			typename cond_variable_type = condition_variable_win, 
			typename TT_ctor = default_ctor<TT>> 

void test_mt_bq(int LOOP_CNT = 10000000, int MAX_THREADS = std::thread::hardware_concurrency()) {

	std::vector<std::thread> threads;

	QueueType<TT, mtx_type, cond_variable_type> queue;

	std::atomic<int> elems = 0;
	std::atomic<int> running_threads = 0;

	for (int ix = 0; ix < MAX_THREADS; ix++) {

		running_threads++;

		threads.push_back(std::thread([&]() {
			auto my_rand = std::bind(std::uniform_int_distribution<int>(0, RAND_MAX), default_random_engine(ix));

			for (int i = 0; i < LOOP_CNT; i++) {

				if (my_rand() % 2) {
					queue.put(TT_ctor()());
					elems++;
				}
				else {
					elems--;
					queue.get();
				}
			}

			running_threads--;

		}));
	}

	// watchdog
	running_threads++;
	threads.push_back(std::thread([&]() {

		while (running_threads.load() > 1) {
			while (elems.load() <= 0) {
				queue.put(TT_ctor()());
				elems++;
			}
			
			this_thread::yield();
		}

		running_threads--;

	}));

	auto last_consumer = std::thread([&running_threads, &elems, &queue]() {

		while (running_threads.load() > 0 || elems.load() > 0) {
			while (elems.load() > 0) {
				queue.get();
				elems--;
			}
			this_thread::yield();
		}
	});


	std::for_each(threads.begin(), threads.end(), [](auto &th) {
		th.join();
	});

	// just make sure there is at least one element to consume, so the thread above does not block.

	elems++;
	queue.put(TT_ctor()());
	last_consumer.join();
}
```

This functio has several parts:
- It creates `MAX_THREADS` which randomly produce and consume events from the queue
- It creates a watchdog thread to make sure the `MAX_THREADS` do not all starve and lock
- It creates a consumer thread just to make sure the queue does not grow too much.

Initially I have started without this consumer thread and I realized that, at the function exit, there are around 3000 unconsumed elements in the queue when considering 1000000 loops in each of the `MAX_THREADS` (monitored in the `elems` variable)

### Results from the simple queue testing:

1. Around 80% of the time it spends in the `mutex::lock` function, in kernel. Optimizing the amount of locking brings by far the biggest performance improvements (I used Visual Studio performance analysis tools).
2. The `std::mutex` does, by default, more spinning than the defaults in Windows `CRITICAL_SECTION`. When using the `std::mutex`, the CPU goes to 100% and the system becomes almost unresponsive. When using the Windows `CRITICAL_SECTION`, with the default spin count, the CPU stays at around 70% (configuration: 4 cores, MAX_THREADS = 4, no `last_consumer` thread). 
3. For 1000000 loops, the amount of memory allocations is about 2500. The `std::deque` is very memory stable and reuses very nicely the memory already allocated on repeated `pop_front`, `push_back`. 

Here is the Windows `CRITIAL_SECTION` and `CONDITION_VARIABLE` implementation of the synchronization interface:

```csharp
class critical_section_win {
private:
	CRITICAL_SECTION cs;
public:
	friend class condition_variable_win;
	std::atomic<int> lk_cnt = 0;

public:

	critical_section_win() {
		// ::InitializeCriticalSectionAndSpinCount(&cs, 0); 
		::InitializeCriticalSection(&cs); // leave the spin count as it is; test the defaults
	}

	~critical_section_win() {
		::DeleteCriticalSection(&cs);
	}

	critical_section_win(const critical_section_win& csw) = delete;
	critical_section_win(critical_section_win&& csw) = delete; // TODO: move

	bool try_lock() {

		int k = 0;
		if (!lk_cnt.compare_exchange_strong(k, 1) )
			return false;

		if (!TryEnterCriticalSection(&cs)) {
			lk_cnt--;
			return false;
		}
		return true;
	}

	void lock() {
		lk_cnt++;
		EnterCriticalSection(&cs);
	}

	void unlock() {		
		LeaveCriticalSection(&cs);
		lk_cnt--;
	}
};

class condition_variable_win {
private:

	CONDITION_VARIABLE cv;
public:

	condition_variable_win() {
		InitializeConditionVariable(&cv);
	}

	condition_variable_win(const condition_variable_win& csw) = delete;
	condition_variable_win(condition_variable_win&& csw) = delete; // TODO: move

	~condition_variable_win() {
		// TODO: defensive programming - test for locked threads
	}

	void notify_all() {
		WakeAllConditionVariable(&cv);
	}

	void notify_one() {
		WakeConditionVariable(&cv);
	}

	void wait(const std::unique_lock<critical_section_win>& csw) {
		SleepConditionVariableCS(&cv, &csw.mutex()->cs, INFINITE);
	}

	template<class R, class P> 
	cv_status wait_for(const unique_lock<critical_section_win>& csw, const chrono::duration<R, P>& duration) {

		DWORD ms = static_cast<DWORD>(chrono::duration_cast<chrono::milliseconds>(duration).count());

		if (!SleepConditionVariableCS(&cv, &csw.mutex()->cs, ms)) {
			return cv_status::timeout;
		}

		return cv_status::no_timeout;
	}
};
```

### Multi-queue blocking queue:

So the main optimization has to do with locking. The simplest way to achieve it is to have a multi-queue implemented as a collection of several other queues and `try_lock` each of them one by one, until the ownership is obtained. In case the ownership cannot be acquired, simply move to the next. With 4 queues and a spin count of 2 (try locking twice all the queues before forcibly locking one of them), the total amount of time spent in synchronization in external code becomes trivial. `mutex::lock` completely loses its predominant status in the list of CPU spent-time and is replaced by other functions like accessing the internal `std::deque` or inserting a new element in it (`get`, `put`). 

Here is the code:

```csharp
template<typename T, typename mtx_type, typename cond_variable_type> class multi_blocking_queue {

private:

	static const int QUEUES = 4; 
	static const int SPIN_COUNT = 2;

	mtx_type				mtx[QUEUES];
	mtx_type				mtx_block;
	cond_variable_type		cv;
	blocking_queue<T, no_lock_mtx, no_lock_condition_var> _queues[QUEUES];

public:

	multi_blocking_queue() {}

	int get_next_queue() {		
		static thread_local int i = 0;
		return i = ((i++) % QUEUES); // round robin for better filling of queues - do not allow elements to remain for too long in one queue
	}

	bool try_get_all_queues(char unallocated[sizeof(T)]) {

		for (int i = 0; i < QUEUES; i++) {

			int q = get_next_queue();

			unique_lock<mtx_type> lk(mtx[q], std::defer_lock);
			if (lk.try_lock() && _queues[q].try_get(unallocated))
				return true;
		}

		return false;
	}

	T get() {

		char stack_alloc[sizeof(T)];
		T* ret = reinterpret_cast<T*>(stack_alloc); // do not call constructor here.

		while (true) {
			for (int sp = 0; sp < SPIN_COUNT; sp++)
				if (try_get_all_queues(stack_alloc))
					return move(*ret);
						
			unique_lock<mtx_type> lk(mtx_block);
			if (try_get_all_queues(stack_alloc))
				return move(*ret);

			cv.wait(lk);
		}
	}

	bool try_put_all_queues(T&& t) {

		for (int i = 0; i < QUEUES; i++) {

			int q = get_next_queue();

			unique_lock<mtx_type> lk(mtx[q], std::defer_lock);
			if (lk.try_lock()) {
				_queues[q].put(move(t));
				return true;
			}
		}

		return false;
	}

	void put(T&& t) {
		__put(move(t));
		cv.notify_one();
	}

	void __put(T&& t) {

		for (int sp = 0; sp < SPIN_COUNT; sp++) {
			if (try_put_all_queues(move(t)))
				return;
		}
	
		{
			unique_lock<mtx_type> lk_block(mtx_block);
			while (!try_put_all_queues(move(t)));
		}
	}
};
```

Notes on the implementation:

- To keep the `try_lock` semantic as described for the `blocking_queue`, I had to trick the underlying `blocking_queues` into no locking. For this specific purpose I have created two clases:

```csharp
class no_lock_mtx {
public:
	void lock() {}
	void unlock() {}
};

class no_lock_condition_var {
public:
	void notify_all() {}
	void notify_one() {}
	void wait(const std::unique_lock<critical_section_win>& csw) {}
};
```
and passed them as synchronization primitives to the underlying `blocking_queues`.
