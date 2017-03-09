---
layout: post
title:  "C++ Play - Green Threads in C++"
date:   2017-03-09 16:15:16 +0200
categories: Native coding
---

In this post I will talk about [green threads](https://en.wikipedia.org/wiki/Green_threads) and give an example implementation in C++ - or, better say, in ASM. The code is highly compiler-dependant (VC++17) and fragile, thus not production-ready. It makes assumptions about stack layout and data in registers and even something as simple as turning on optimizations will most likely crash it. But it was a fun programming exercise for a concept often found in asynchronous libraries or implemented directly in language constructs. Interesting links: [Fibers](https://en.wikipedia.org/wiki/Fiber_%28computer_science%29), [Coroutines](https://en.wikipedia.org/wiki/Coroutine), [Set Context](https://en.wikipedia.org/wiki/Setcontext), [Actors](https://en.wikipedia.org/wiki/Actor_model), [Cooperative multitasking](https://en.wikipedia.org/wiki/Cooperative_multitasking)

### The problem

You have N CPUs and M threads, where M >> N  - the case for actors, for instance. You know that these M threads will most likely interact with each other (either passing messages or synchronizing on some primitive). You might also have I/O and you want to keep the program flow simple and not pollute it with lots of callbacks - synchronous I/O code is clearly much simpler to read than async I/O, but also more heavy on the OS. These problems are suitable for considering a cooperative scheduling approach.

### The API and the test scenario

For this demo I will only implement `wait-for` (waiting for a thread to finish) and `yield` (give control to the next thread). Any other operations would only make the code more complex to read but not add much clarity into the concepts. Here is my test bed:

![vs_run]({{site.ur}}/assets/green_threads_1.png)

```csharp
void fn(thread_pool::thread_ctx *ctx, int p1, int p2) {

	if (p1 >= 4) {
		cout << "Ended NO yields: " << ctx->name() << endl;
		return;
	}
	
	char thread_name[80];
	sprintf_s(thread_name, "Thread %d - %d", p1 + 1, p2 + 2);
	cout << "Starts: " << thread_name << endl;

	auto child1 = ctx->call_fn(100000, thread_name, fn, p1 + 1, p2 + 2);

	sprintf_s(thread_name, "Secondary thread %d - %d", p1 + 1, p2 + 2);
	cout << "Starts: " << thread_name << endl;

	auto child2 = ctx->call_fn(100000, thread_name, fn, p1 + 1, p2 + 2);

	std::vector<decltype(child1)> children;

	for (int i = 0; i < 7; i++) {

		cout << p1 << p2 << endl;

		auto p = ctx->call_fn(10000, "Child of child", fn, p1 + 1, p2 + 2);

		children.push_back(p);
		ctx->yield();
	}

	ctx->wait_for(child1);
	ctx->wait_for(child2);

	for (auto &c : children)
		ctx->wait_for(c);

	cout << "Ended WITH yields: " << ctx->name() << endl;
}

#pragma optimize ("", on)
int main()
{
	int p1 = 0;
	int p2 = 0;

	thread_pool pool;
	pool.call_fn(100000, "THREAD_0" ,fn, p1, p2);	
	return 0;
}

```

In main I simply call on the main thread, which I name "THREAD_0" a procedure, `fn`, defined above. This is (and should be) a synchronous call. `fn` should wait for all spawned threads to be finished. Note: when I say "thread", I mean a "green thread". All my threads share the same OS thread. In this sample there is no OS multithreading involved. From the OS perspective, the application is single threaded. 

The `fn` function spawns recursively many other child-threads which are waited for at the end of the function. Context switching happens when control reaches the `ctx->yield()` call. `ctx` is our current thread (in code called `thread_context`). If control does not reach a `ctx->yield()`, the function is simply executed on the stack of the caller, synchronously, like a normal function. When control reaches `ctx->yield()` the function will be put on hold and the rest of the threads will be executed. On the spawning thread, the function seems to return asynchronously. The result is a `thread_ctx` object which can be queried for thread completion or waited for. When a thread is spawned, parameters can be sent to it like to any other function, on the stack - the implementation is based on a variadic template. In our case I send two `ints`. 

### Api description:

`template<class fn, typename... T> void thread_pool::call_fn(unsigned int stack_size, const char* name, fn* f, T... params)` - creates the parent thread. It is the way to initialize the threading library because for the parent thread return information should be stored in the main process stack. The function should wait for all its children to finish before exiting. 

`template<class fn, typename... T> shared_ptr<thread_ctx> thread_pool::thread_ctx::call_fn(unsigned int stack_size, const char* name ,fn* f, T... params)` - spawns a new child thread. 

`void thread_pool::thread_ctx::wait_for(shared_ptr<thread_ctx>& other)` - waits for a child. 

`void thread_pool::thread_ctx::yield()` - gives control to the next thread waiting for execution in a round-robin fashion. 

### Implementation




