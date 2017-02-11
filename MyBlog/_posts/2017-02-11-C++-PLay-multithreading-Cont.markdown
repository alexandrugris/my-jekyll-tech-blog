---
layout: post
title:  "C++ Play - Multithreading Part 2"
date:   2017-02-04 12:15:16 +0200
categories: Native coding
---
In this post I am going to touch a little bit synchronization primitives and write two different implementations for a reader-writer lock. The code comes from an earlier project, before `std::shared_lock` was introduced to the standard. 
Beside the standard library, RW locks are part of Windows SDK as [Slim Reader/Writer (SRW) Locks](https://msdn.microsoft.com/en-us/library/windows/desktop/aa904937(v=vs.85).aspx) and on Linux, with [pthreads](https://linux.die.net/man/3/pthread_rwlock_rdlock).
There is no need (and not recommended) to implement them manually, as I do in the code below. But since this is a tech blog and and it is meant to play around with technology, here are two possible implementations.

### RW-Lock with CAS

CAS stands for compare-and-swap. These are primitives which translate directly to single assembly instructions and are meant for atomic operations on integral data types. Being implemented directly in hardware, 
they are the building blocks for constructing all synchronzation primitives, as well as non-locking algorithms and data structures. A non-locking algorithm is an algorithm which guarantees safe multithreaded operations without need to make a kernel call. 
For short operations they should be faster, but speed comes also at a price: they cannot be used in single CPU systems because they basically spin the CPU in a tight loop and, for battery-powered devices, they tend to heat and drain the battery.

In the standard library, CAS is present through [`std::atomic`](http://en.cppreference.com/w/cpp/atomic/atomic). 

### First, the usage and some foundations

I would like to be able to use `std::unique_lock` with my library as it might make my code usable with already existing templates which receive the lock as a template parameter. 
In my code I played with various types of mutexes to test performance, so `MutexType` below is a template parameter. I am going to use the following constructs:

```csharp
MutexType mtx;

// write lock
std::unique_lock<MutexType::write_mutex> lck(mtx);
```

and 

```csharp
MutexType mtx;

// read lock
std::unique_lock<MutexType::read_mutex> lck(mtx);
```


In order to build my mutex I need two base classes: `read_mutex` and `write_mutex` from which to derive and static cast later.

```csharp
template<class T> class read_lockable{
public:
    
    typedef read_lockable<T> read_mutex;
    
    // these methods are called by unique_lock
    void lock() { static_cast<T*>(this)->lock_read(); }
    void unlock() { static_cast<T*>(this)->unlock_read(); }
};

template<class T> class write_lockable{
public:
    
    typedef write_lockable<T> write_mutex;
    
    void lock() { static_cast<T*>(this)->lock_write(); }
    void unlock() { static_cast<T*>(this)->unlock_write(); }
};
```

### First implementation - pure CAS, non-blocking:

```csharp
class rw_spinlock : public read_lockable<rw_spinlock>, 
					public write_lockable<rw_spinlock>{    
public:
        
    rw_spinlock() : reads(0), writes(0){}
    
    ~rw_spinlock(){
#ifdef _DEBUG
        if(!(reads == 0 && writes == 0)){ 
			__debugbreak();
		}
#endif
    }

private:

    // this is completely from me. I don't know if it makes sense
    // it basically tries to spin for a number of cycles in tight loop before it
    // attempts to call into the kernel for the next thread to be scheduled
    // TODO: IF production code, add performance counters and play around with variables
	const int SPIN_COUNT_MAX = 15000;
	void spin(int & spn) {
		if (--spn <= 0) {
			std::this_thread::yield();
			spn = 0;
		}
	}

public:
    
    void lock_read(){
		int spn = SPIN_COUNT_MAX;

        // wait for writesto finish	
        int wr = 0;
        while(!writes.compare_exchange_weak(wr = 0, 1)){
			spin(spn);
        }
        reads++;
        writes.store(0);
    }
    
    void lock_write(){
		int spn = SPIN_COUNT_MAX;
        
        // acquire the writes lock first
        // so that no new reads or writes can enter the loop
        int wr = 0;
        while(!writes.compare_exchange_weak(wr = 0, 1)){
			spin(spn);
        }
        // wait for finishing the reads
        int rd = 0;
        while(!reads.compare_exchange_weak(rd = 0, 0)){
			spin(spn);
		}
    }
    
    void unlock_read(){ 
		reads--;
	}
    
    void unlock_write(){ 
#ifdef _DEBUG
		if (!(reads == 0 && writes == 1)){
			__debugbreak();
		}
#endif
		writes--; 		
	}
    
private:
    std::atomic<int> reads;
    std::atomic<int> writes;
};
```

### Second implementation - CAS only on read, block with mutex the writes:


```csharp    
class rw_mutex :
    public read_lockable<rw_mutex<count_locks>>,
    public write_lockable<rw_mutex<count_locks>>,
   {
        
public:
    
    rw_mutex(){}

private:    
    std::mutex mtx;
    std::condition_variable unlock_reads;

public:
    
    void lock_read(){		
        std::unique_lock<std::mutex> lk(mtx);
        int wr = 0;
        while (!writes.compare_exchange_weak(wr = 0, 0)) {
            unlock_reads.wait(lk);  
        }
        reads++;		
    }
       
    void lock_write(){

		mtx.lock();
		writes++; // stop accepting the reads

		// wait for all reads to finish
		int rd = 0;
		while (!reads.compare_exchange_weak(rd = 0, 0)) {
			std::this_thread::yield();
		}
    }

	void unlock_read(){
		reads--;
	}

	void unlock_write(){		
		writes--;
		mtx.unlock(); 

		unlock_reads.notify_all();
	}
	
private:		
	std::atomic<int> writes = 0;
	std::atomic<int> reads = 0;
};

```

### CAS operations used:

1. Atomic increment and decrement `std::atomic<int>::operator++()` and `std::atomic<int>::operator--()`
2. Compare exchange weak. From the [documentation](http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange):

>Atomically compares the object representation of *this with the object representation of expected, as if by std::memcmp, and if those are bitwise-equal, replaces the former with desired (performs read-modify-write operation). 
>Otherwise, loads the actual value stored in *this into expected >>(performs load operation). Copying is performed as if by std::memcpy. 
>The memory models for the read-modify-write and load operations are success and failure respectively. In the (2) and (4) versions order is used for both read-modify-write and load operations, 
>except that std::memory_order_acquire and std::memory_order_relaxed are used for the load operation if order == std::memory_order_acq_rel, or order == std::memory_order_release respectively. 

and

>The weak forms (1-2) of the functions are allowed to fail spuriously, that is, act as if *this != expected even if they are equal. 
>When a compare-and-exchange is in a loop, the weak version will yield better performance on some platforms. 
>When a weak compare-and-exchange would require a loop and a strong one would not, the strong one is preferable 
>unless the object representation of T may include padding bits, trap bits, or offers multiple object representations for the same value (e.g. floating-point NaN). 
>In those cases, weak compare-and-exchange typically works because it quickly converges on some stable object representation. 

### A note on performance:

The code above comes from playing around building a configurable in-process multithreaded cache. I measured performance of various locking primitives and I've noticed that actually the fastest code runs on `std::mutex` directly. 
My tests had randomly distributed equally numbered reads and writes, so the reader-writer optimization did not make much of a difference. `std::mutex` is implemented on top of the OS mutex which is alread highly optimized. It also uses 
CAS primitives internally to spin a little before the actual kernel call, but in their case the number come from a wide variety of tests.
Therefore, my recommendation is not to try to build custom-made synchronization primitives to improve performance. They are fun, however, as programming practice. 