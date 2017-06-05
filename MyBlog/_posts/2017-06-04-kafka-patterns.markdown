---
layout: post
title:  "Apache Kafka"
date:   2017-06-04 13:15:16 +0200
categories: distributed systems
---
Playing around with Apache Kafka.

### How to run Apache Kafka

The simplest way: `docker-compose up` with the following `docker-compose.yml`

```yml
version: '2'

# based on this https://github.com/wurstmeister/kafka-docker

services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181:2181"

  kafka:
    image: wurstmeister/kafka

    ports:
      - "9092:9092"
      - "1099:1099" # for JMX

    environment:
      KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_JMX_OPTS: "-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Djava.rmi.server.hostname=127.0.0.1 -Dcom.sun.management.jmxremote.rmi.port=1099"
      JMX_PORT: 1099

    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

Another option, if we want to scale the Kafka cluster is to run:

```
docker-compose up -d
```

and then

```
docker-compose scale kafka=2
```

However, in order for this to run, we need to make some changes to the docker-compose file:

```
version: '3'

# based on this: https://github.com/wurstmeister/kafka-docker

services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181:2181"

  kafka:
    image: wurstmeister/kafka

    ports:
      - "9092" # we do not publish these ports anymore to host

    environment:
      HOSTNAME_COMMAND: "route -n | awk '/UG[ \t]/{print $$2}'"
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

In this docker image, Kafka is installed in `/opt/kafka`, thus running

```
docker exec testkafka_kafka_2 ls /opt/kafka/bin
```

will get get us the list of all Kafka binaries.

### Basic Concepts

Apache Kafka is a high-throughput distributed pub-sub messaging system, with on-disk persistence. In essence, it can be viewed as a distributed immutable log, where messages are appended to the end of the log and each client has a read-only cursor for reading.

Vocabulary: 

- A Kafka cluster is a grouping of multiple Kafka brokers (could be hundreds)
- Work (in this case message submission, categorization and storage) is assigned to the cluster by a Producer and is consumed by a Consumer. Inside the cluster we have 3 node categories: Controller (one per cluster), Leader (one per task / group of nodes) and Followers. 




