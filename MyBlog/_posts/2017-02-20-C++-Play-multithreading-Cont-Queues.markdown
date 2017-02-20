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


