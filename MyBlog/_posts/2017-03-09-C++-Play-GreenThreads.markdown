---
layout: post
title:  "C++ Play - Green Threads in C++"
date:   2017-03-09 16:15:16 +0200
categories: Native coding
---

In this post I will talk about [green threads](https://en.wikipedia.org/wiki/Green_threads) and give an example implementation in C++ - or, better say, in ASM. The code is highly compiler-dependant (VC++17) and fragile, thus not production-ready. It makes assumptions about stack layout, data in registers, and even turning on optimizations will most likely crash it. But it was a fun programming exercise for a concept often found in asynchronous libraries or implemented directly in language constructs. Interesting links: [Fibers](https://en.wikipedia.org/wiki/Fiber_%28computer_science%29), [Coroutines](https://en.wikipedia.org/wiki/Coroutine), [Set Context](https://en.wikipedia.org/wiki/Setcontext), [Actors](https://en.wikipedia.org/wiki/Actor_model), [Cooperative multitasking](https://en.wikipedia.org/wiki/Cooperative_multitasking)



