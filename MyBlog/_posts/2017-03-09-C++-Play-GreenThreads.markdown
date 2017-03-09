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

Here is the full listing:

```csharp
#define _DEBUG

#pragma optimize ("", off)

class thread_pool {

private:
	unsigned int current_thread = 0;	

public:

	struct thread_ctx {

	private:
		char* thread_name = nullptr;

	private:
		unsigned char* stack = nullptr;
		unsigned char* stack_ptr = nullptr;
		unsigned char* continuation_location = nullptr;

		thread_pool* th_p = nullptr;
		bool	b_finished = false;
		bool	b_yielded = false; // useful for when we have an out of order return from the function.

#ifdef _DEBUG
		int stack_size = 0;
#endif

	public: 
		friend class thread_pool;
		thread_ctx(thread_pool* tp = nullptr, int _stack_size = -1) : th_p(tp){

#ifdef _DEBUG
			stack_size = _stack_size;
#endif
			stack = new unsigned char[_stack_size];
			stack_ptr = stack + _stack_size;
		}

		~thread_ctx() {
			delete[] stack;
			// cout << "Deleted: " << thread_name << endl;
			if(thread_name)
				free(thread_name);
		}

		thread_pool* get_thread_pool() { return th_p; }

		const char* name() const { return thread_name; }

		bool finished() const { return b_finished; }

		void wait_for(shared_ptr<thread_ctx>& other) {
			while (!other->b_finished)
				yield();
		}

		void yield() {

			int stk_tst = 0;
			b_yielded = true;

#ifdef _DEBUG
			// test we are on our stack
			assert( int(&stk_tst) >= int(stack) && int(&stk_tst) < stack_size + int(stack));
#endif
			
			unsigned char** loc_ptr		= &continuation_location;
			auto next					= th_p->next();

			while (next->b_finished || (next.get() != this && next->continuation_location == nullptr))
				next = th_p->next();

			auto next_ptr = next.get();

			// cout << "Yield: From " << this->thread_name << " To " << next_ptr->thread_name << endl;

			// save the jump location for this thread
			__asm {				
				mov eax, offset continuation_code_ptr;
				mov ebx, loc_ptr;
				mov [ebx], eax;
			}

			// save the current stack_ptr to restore it
			__asm{
				mov ecx, this;
				push stk_tst;
				push ebp;
				mov [ecx + stack_ptr], esp;
			}			
			
			// do the jump to to next location,
			__asm {
				mov ecx, next_ptr;
				mov ebx, [ecx + continuation_location]; // next is in ecx
				mov esp, [ecx + stack_ptr];
				jmp ebx;			
			}

			assert(0); // should never get here
			
			// jmp location:
			__asm
			{
			continuation_code_ptr:
				pop ebp;
				pop stk_tst; // should be 0;
			}

			assert(stk_tst == 0);			
		}

		template<class fn, typename... T> shared_ptr<thread_ctx> call_fn(unsigned int stack_size, const char* name ,fn* f, T... params) {
			auto ctx = make_shared<thread_ctx>(th_p, stack_size);

			if (name) 
				ctx->thread_name = _strdup(name);

			th_p->threads.push_back(ctx);
			ctx->assign_fn(this, f, params...);
			return ctx;
		}

	private:

		template<class fn, typename... T> void assign_fn(thread_ctx* parent_thread, fn* f, T... params) {
			int stk_tst = 0;

			unsigned char* stk = stack_ptr;	

			if (parent_thread == nullptr) { // we are on root of threads
				// setup start of new thread
				__asm {
					mov eax, esp; // save old stack on new stack
					mov esp, stk;
					push eax;
				}

				f(this, params...);

				__asm {
					pop esp;
				}

				b_finished = true;
				th_p->remove(this);
			}
			else {
				unsigned char** loc_ptr		= &parent_thread->continuation_location;
				unsigned char*	stck_ptr	= nullptr;
				
				// save the jump location -> in theory this is known at compile time, but maybe there will be different jump points
				__asm {
					mov eax, offset continuation_code_ptr;
					mov ebx, loc_ptr;
					mov[ebx], eax;
				}

				// save the current stack_ptr to restore it (we are on parent stack)
				__asm {
					push stk_tst;
					push ebp;
					mov[stck_ptr], esp;
				}

				parent_thread->stack_ptr = stck_ptr;

				// switch to the new thread stack
				__asm {
					mov esp, stk;
					push stck_ptr; // these still work because we have not changed ebp
					mov eax, this;
					push eax;
				}

				f(this, params...);

				__asm {

					pop ecx; // this is in ecx
					mov eax, [ecx + b_yielded];
					and al, 1
					jnz function_already_returned_once;
				}

				// switch back to parent stack
				// restore this
				__asm {
					pop eax; // stk_ptr
					mov esp, eax; // stk_ptr
					pop ebp;					
					pop stk_tst; // here we already have the values
				}

				assert(stk_tst == 0);

				b_finished = true;
				th_p->remove(this);

				return;

				__asm {
				function_already_returned_once:

					// this is in ecx
					mov eax, [ecx + b_finished];
					or al, 1;
					mov[ecx + b_finished], eax; // set the b_finished flag to true

					mov ebx, ecx; // save this temporary
					mov ecx, [ebx + th_p];

					push ebx; // once for saving, once for remove call
					push ebx;
					call remove;

					pop ecx;
					call yield

					// TODO: set the b_finished flag and then remove the thread
					// do yield
				}

				assert(0); // should never get here.

				__asm
				{
				continuation_code_ptr: // only for jmp code from another place
					pop ebp;
					pop stk_tst; // should be 0;
				}

				assert(stk_tst == 0);
				// cout << "Async return for " << this->thread_name << endl;
			}

		}			
	};


private:
	std::vector<shared_ptr<thread_ctx>> threads;

public:

	shared_ptr<thread_ctx> next() {
		for( unsigned th = 0; th < threads.size() ; th ++){
			auto ret = threads[(++current_thread) % threads.size()];
			if (ret != nullptr)
				return ret;
		} 

		return nullptr;
	}

	void remove(const thread_ctx* ctx) {
		for(unsigned int i = 0 ; i < threads.size(); i++)
			if (threads[i].get() == ctx) {
				threads[i] = threads[threads.size() - 1];
				threads.pop_back();
				return;
			}
		assert(0); // not found
	}

	~thread_pool() {
		for (unsigned int i = 0; i < threads.size(); i++) {
			threads[i] = nullptr;			
		}
	}

	template<class fn, typename... T> void call_fn(unsigned int stack_size, const char* name, fn* f, T... params) {
		auto ctx = make_shared<thread_ctx>(this, stack_size);

		if(name) ctx->thread_name = _strdup(name);

		threads.push_back(ctx);
		ctx->assign_fn(nullptr, f, params...);
	}
};

#pragma optimize ( "", off)

// spanwn a huge tree of children.
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

And the explanations:

```csharp
struct thread_ctx {

	private:
		char* thread_name = nullptr;

	private:
		unsigned char* stack = nullptr;
		unsigned char* stack_ptr = nullptr;
		unsigned char* continuation_location = nullptr;

		thread_pool* th_p = nullptr;
		bool	b_finished = false;
		bool	b_yielded = false; // useful for when we have an out of order return from the function.
```

 - `thread_ctx` - our thread. 

