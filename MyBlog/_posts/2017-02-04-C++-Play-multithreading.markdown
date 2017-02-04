---
layout: post
title:  "C++ Play - Multithreading"
date:   2017-02-04 12:15:16 +0200
categories: Native coding
---
In this post I am planning to build upon C++ 11/14 features introduced in the previous posts and play around with threads and synchronization. 
I am going to start with the threading support introduced in the standard library then move forward to Windows-specific code. Unix multithreading and server programming I will probably cover in another post.

### C++ standard library and threads

The new C++ standard adds [extensive support](http://en.cppreference.com/w/cpp/thread) for thread management as well as a new storage specifier, `thread_local` ([storage specifiers](http://en.cppreference.com/w/cpp/language/storage_duration)).

Let's start with an interesting piece of code which combines many of these concepts together:

```csharp
class TestMTLocalStorage {
	int i = 0;
public:

    // called only once
	TestMTLocalStorage(int _i) : i(_i) { cout << "Constructor" << endl; } 

    // not called
	TestMTLocalStorage(const TestMTLocalStorage& t) {
		cout << "Copy " << endl;
	}

    // not called
	TestMTLocalStorage() { cout << "default" << endl; } // not called

    // not called
	TestMTLocalStorage(TestMTLocalStorage&& mv) noexcept { cout << "move" << endl; }

	operator int&() { return i; }

    // called after exit from main, similar to static variables, only once
	~TestMTLocalStorage() noexcept { 
		cout << "Destructor" << endl;  
	}
};

int compute_i() {
	cout << "Compute i called" << endl;
	return 0;
}

void test_multithreading_thread_local() {

	thread_local TestMTLocalStorage i = compute_i(); // can be initialized dinamically;

	mutex m;
	condition_variable cv;

	cout << "Address of i is: 0x" << hex << &i << endl;

	auto fn = [&m, &cv]() {
		{
			unique_lock<std::mutex> ul(m);
			cv.wait(ul);
		}

		for (int j = 0; j < 100000; j++)
			i++; // some computation;

		{
			lock_guard<mutex> lck(m);
			cout << "Thread local variable i is: " << dec << i 
                    << ". Address of i is: 0x" << hex << &i << endl; 
                    // different addresses of i;
		}

	};

	thread t1(fn);	
	thread t2(fn);

	cv.notify_all();

	t2.join();
	t1.join();

}

int main()
{	
	test_multithreading_thread_local();

    return 0;
}
```

And here is the output:

![Console output from the code above]({{site.url}}/assets/cpp_multithreading_console.png)

### Thread Local Storage

Running the code we notice the following:

- `thread_local TestMTLocalStorage i = compute_i()` is called as expected. At this moment, the member variable `i` has the `0x013A46A4` address.
- Within the thread, the computations on `i` are independent. Internal variable `i` has different address in each thread, and different from the initial address on which it was initialized.
- No new constructor / destructor is called when entering or exiting the thread.
- Variable `i` is only destructed at the end of the code, after exiting main, the destructor being called only once (at least in VC++ 2017). 

This lifecycle requires special care when managing dynamically allocated memory within a `thread_local` class instance.

### Threading support library (std)

In the code above we use the following:

- Creating and running threads, using the `std::thread` standard library class. Threads can be instantiated with a lambda and can capture any members. In our case, we capture the synchronization primitives `m` and `cv` by reference.
- Thread synchronization in three different scenarios

#### Having all threads wait until a certain condition is met. 

This is done through the following calls: 

```csharp
{
	unique_lock<std::mutex> ul(m);
	cv.wait(ul);
}
```
in the waiting threads and then notifying all threads using the same condition variable ```	cv.notify_all();```. Important to notice that `cv.wait()` requires a `unique_lock` on a mutex in order for the thread to block.

#### Protecting critical sections of code from running in parallel

Using the scoped:

```csharp
{
    lock_guard<mutex> lck(m);
    cout << "Thread local variable i is: " << dec << i 
         << ". Address of i is: 0x" << hex << &i << endl; 
        // different addresses of i;
}
```
the protection being against several threads running `cout` in parallel and generating garbage output on the console. As we have already seen, `i` is `thread_local`.

#### Waiting for several threads to finish before exiting the method.

```csharp
t2.join();
t1.join();
```

### Other interesting concepts from the standard threads library:

- `recursive_mutex` - allows the same thread to lock the mutex several times. Mutex is released when the number of locks matches the number of releases.
- `shared_mutex` - mutex that can be used for reader-writer scenarios. Several threads can have shared ownership (read) while only one thread can have exclusive ownership (write).
- `scoped_lock` - tries to take ownership of several mutexes.
- `future`, `promise`, `async` - futures package to facilitate asynchronous programming, without manual management of threads - [async example](http://en.cppreference.com/w/cpp/thread/async).














