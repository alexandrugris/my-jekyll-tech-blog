---
layout: post
title:  "Spark Play"
date:   2017-12-10 13:15:16 +0200
categories: big data

```
c:>docker run --rm -it -p 4040:4040 -v "C:\Users\alexa\Downloads\DataScience - RecordLinkage:/data"  gettyimages/spark bin/spark-shell --driver-memory 3g
```
```
val csv = spark.read.option("header", "true").option("nullValue","?").option("inferSchema", "true").csv("/data")
csv.printSchema()
```

TODO: chaching options MEMORY,MEMORY_SER,MEMORY_AND_DISK, MEMORY_AND_DISK_SER



