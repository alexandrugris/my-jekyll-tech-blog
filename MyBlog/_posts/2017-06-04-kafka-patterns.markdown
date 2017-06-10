---
layout: post
title:  "Apache Kafka"
date:   2017-06-04 13:15:16 +0200
categories: distributed systems
---
Playing around with Apache Kafka. The article covers running a Kafka cluster on a development machine using a pre-made Docker image, playing around with the command line tools distributed with Apache Kafka and writing producers and consumers.

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
      - "1099:1099" # for JMX - not needed unless we want to monitor the service

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

```yml
version: '3'

# based on this: https://github.com/wurstmeister/kafka-docker

services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181"

  kafka:
    image: wurstmeister/kafka

    ports:
      - "9092" # we do not publish these ports anymore to host
      
    environment:
      HOSTNAME_COMMAND: "route -n | awk '/UG[ \t]/{print $$2}'"
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      # we dropped completely the JMX part

    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

In this docker image, Kafka is installed in `/opt/kafka`, thus running

```
docker exec testkafka_kafka_2 ls /opt/kafka/bin
```

will get get us the list of all Kafka binaries. Let's connect to one of the two Kafka instances.

```
docker exec -it testkafka_kafka_2 bash
```

![Kafka Connection]({{site.url}}/assets/kafka_1.png)

However, if JMX is specified, the `JMX_PORT` is already set and in use by the Kafka daemon so in order to run commands to Kafka we need to make a little hack:

```
$>JMX_PORT=1100
$>KAFKA_JMX_OPTS=""
```

Then we can play:

![Kafka Commands]({{site.url}}/assets/kafka_2.png)

To produce messages to `my_topic`:

```
bash-4.3# ./kafka-console-producer.sh --topic my_topic --broker-list testkafka_kafka_1:9092
```

To read messages from `my_topic`:

```
bash-4.3# ./kafka-console-consumer.sh --topic my_topic --zookeeper zookeeper:2181 --from-beginning
```

### Basic Concepts

Apache Kafka is a high-throughput distributed pub-sub messaging system, with on-disk persistence. In essence, it can be viewed as a distributed immutable ordered (by time) sequence of messages. Each consumer has a private read-only cursor, which it can reset at any time. This way of storing messages / events makes Kafka appropriate as a distributed store for event sourcing architectural pattern.

*Vocabulary:*

- A Kafka cluster is a grouping of multiple Kafka brokers (could be hundreds).
- Topics can span over a multitude of brokers.
- Each topic has one or more partitions.
- Each partition is maintained over one or more brokers. 
- Each partition must fit on one machine.
- Each partition is mutually exclusive from each other. This means the same message wil never appear simultaneously on two partitions.
- If there is no partitioning scheme set up, messages are put to partitions in a round-robin fashion.
- Message order is guaranteed only on a per-partition basis.

*Partitioning trade-offs:*

- The more partitions the more load on Zookeeper. Since Zookeeper keeps its data in memory only, resources on the Zookeeper machine can become constrainted.
- There is no global order in a topic with several partitions. 
- As each partition can only fit on a single machine. This imposes the limit to how much a single partition can grow.

*Fault tolerance:*

 - Broker failure
 - Disk failure
 - Network failure

 Replication factor is set on a topic level, thus can vary from topic to topic but not from partition to partition.

 When we connect to both docker instances that we started in our demo cluster, because we set for topic `my_topic` 2 partitions and replication factor 2, we see the same picture:

 ![One topic, 2 partitions, replication factor 2]({{site.url}}/assets/kafka_3.png)

 In the image above, `ISR` means in-sync replicas. If we decide to kill one docker container, we get:

 ```
bash-4.3# /opt/kafka/bin/kafka-topics.sh --describe my_topic --zookeeper zookeeper:2181
Topic:my_topic  PartitionCount:2        ReplicationFactor:2     Configs:
        Topic: my_topic Partition: 0    Leader: 1001    Replicas: 1002,1001     Isr: 1001
        Topic: my_topic Partition: 1    Leader: 1001    Replicas: 1001,1002     Isr: 1001
 ``` 

 The cluster has readjusted itself. Leader for partition 0 has been assigned 1001 (instead of 1002 which was previously), we still have two replicas, but ISR is only 1001.

 If we start the stopped container again, we get:

```
bash-4.3# /opt/kafka/bin/kafka-topics.sh --describe my_topic --zookeeper zookeeper:2181
Topic:my_topic  PartitionCount:2        ReplicationFactor:2     Configs:
        Topic: my_topic Partition: 0    Leader: 1001    Replicas: 1002,1001     Isr: 1001,1002
        Topic: my_topic Partition: 1    Leader: 1001    Replicas: 1001,1002     Isr: 1001,1002
 ```

Leader for both partitions stay the same, but this time ISR are both 1001 and 1002.

[How many topics can be created in Apache Kafka?](https://www.quora.com/How-many-topics-can-be-created-in-Apache-Kafka/answer/Jay-Kreps?srid=55SS)

*Retention policy:*

Retention is regardless of whether any consumer has read any message. Default retention time is 7 days. Unlike RabbitMQ for instance, where TTL is set on a per-message basis, in Kafka retention policy is set on a per-topic basis. 

### Debugging Kafka Clients

There are two basic ways to debug Kafka clients when you want to run everything on your machine. The simplest way is to run a single kafka instance and map its port to localhost. Thus, in the client application, there will be only one Kafka broker to connect to, that is `localhost`. 

However, there is another one, slightly more complex but  more rewarding as one can scale as many brokers as he/she wishes. This requires the creation of another service in the docker compose file which runs the Java application to debug, because it is needed that this app runs in the same docker network as the rest of the cluster. It also requires remote debugging enabled and mapping of the project directory in a volume in the docker container that runs the app. Here is the updated `docker-compose.yml` as it is configured on my machine.

```yml
version: '3'

# based on https://github.com/wurstmeister/kafka-docker

services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181"

  kafka_client_app: # the container to which my java client connects to
    image: openjdk

    ports:
      - "8000:8000" # expose the debugger port to localhost

    depends_on:
     - kafka

    command: java -agentlib:jdwp=transport=dt_socket,server=y,address=8000,suspend=y -Dbootstrap.servers=kafka:9092 -cp /myapp/out/production/TestKafka:/myapp/lib/* ro.alexandrugris.BasicKafkaProducer # wait for debugger in suspend mode

    volumes:
      - C:\Users\alexa\IdeaProjects\TestKafka:/myapp # map the project to the container

  kafka: # for scale
    image: wurstmeister/kafka
    depends_on:
      - zookeeper

    ports:
      - "9092"

    environment:
      HOSTNAME_COMMAND: "route -n | awk '/UG[ \t]/{print $$2}'"
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181

    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

And some screenshots:

*Debugger Configuration in Idea*

![Debugger Config Idea]({{site.url}}/assets/kafka_4.png)

*Breakpoint in remote debugging config*
![Debugger Breakpoint]({{site.url}}/assets/kafka_5.png)

*Cluster up, with bash started on app node*
![Cluster]({{site.url}}/assets/kafka_6.png)

*Produced messages in the queue*
![Messages]({{site.url}}/assets/kafka_7.png)

