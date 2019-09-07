---
layout: post
title:  "Live Streaming Architecture Using Apache Kafka And Redis"
date:   2018-10-20 13:15:16 +0200
categories: programming
---

This topic describes a streaming architecture based on Apache Kafka and Redis that can be applied to build high perfoming, resilient streaming architectures.

### Apache Kafka

Apache Kafka allows to store the incoming data stream and computation results used for later stages in the pipeline in a fault tolerant, distributed way. Apache Kafka also allows brining new servers to the system in case of high data load.

- In case of replication, each partition will have one server as the leader and configurable other as followers. Leader is managing read / write requests for a specific partition, while followers are managing replication of the leader.
- In Kafka, leadership is defined per partition. That means that a server can be leader for a partition but a follower on another.
- Zookeeper stores consumer offsets per topics.

A simple way to use start with Kafka for fun projects at home is to use a docker-compose with a setup similar to the following:

```
version : '3.5'

services:
  zookeeper:
    image: "confluentinc/cp-zookeeper"
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181 

  kafka:
    image: "confluentinc/cp-kafka"
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1

  kafka_rest:
    image: "confluentinc/cp-kafka-rest"
    ports:
      - "8082:8082"
    environment: 
      - KAFKA_REST_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_REST_LISTENERS=http://0.0.0.0:8082
      - KAFKA_REST_HOST_NAME=kafka_rest

networks:
  default:
    name: oda
```

Please note that a race condition in the dockerfile above (Zookeeper starts slower than Kafka) leads Kafka service to fail to start and it must me restarted with `docker-compose scale kafka=1`. Once the service is up, we can test the configuration as follows:

```
docker run --net=oda --rm confluentinc/cp-kafka bash -c "seq 42 | kafka-console-producer --request-required-acks 1 --broker-list kafka:9092 --topic foo && echo 'Produced 42 messages.'"

docker run --net=oda --rm confluentinc/cp-kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic foo --from-beginning --max-messages 42
```

Below, a short node script to simplify playing with kafka console tools by providing default parameters to work with the dockerfile presented above:

```javascript
#!/usr/local/bin/node

let child_process = require('child_process');

let cmds = {
    'topics' : ['kafka-topics', '--zookeeper', 'zookeeper:2181'],
    'produce' : ['kafka-console-producer', '--broker-list', 'kafka:9092'],
    'consume' : ['kafka-console-consumer', '--bootstrap-server', 'kafka:9092']
}

let params = null;

process.argv.forEach((arg) => {
    // first with command
    if (params == null && cmds[arg] != null) {
        let cmd = cmds[arg];
        params = ['run', '--net=oda', '--rm', '-it', 'confluentinc/cp-kafka', ...cmd]
    }
    // add the rest
    if (params != null){
        params.push(arg);
    }
});

let docker = child_process.spawn('docker', params, {stdio: 'inherit'});

docker.on('error', (err) => {
    console.log('Failed to start docker');
});

docker.on('close', (code) => {
    console.log(`Child process exited with code ${code}`);
});
```

This allows easy testing as follows:

 - To create a topic named `test`: `./kafka.js topics --create --topic test --replication-factor 3 --partitions 3`
 - To list the topics: `./kafka.js topics --list`
 - To produce some messages: `./kafka.js produce --topic test`
 - To read the messages from the beginning: `./kafka.js consume --topic test --from-beginning`

 Zookeeper can and should be scaled also. Due the quorum required, the formula for determining the number of Zookeeper instances to run is `2 * F + 1` where `F` is the desired fault tolerance factor.

 To test that indeed data is moving through the system, let's write the code for the Kafka Producer. I will use a very basic NodeJS package which connects to Kafka's REST API. For production use, the more advanced packages like `kafka-node` and the fast growing at the moment of writing this `kafkajs` should be preferred. I use here the `kafka-rest` package for simplicity and convenience. 

 ```javascript
const KafkaRest = require('kafka-rest');

// kafka rest is exposed from docker-compose on 8082
const kafka = new KafkaRest({ 'url': 'http://localhost:8082' });

// make sure the topic is created before
const target = kafka.topic('random_walk');

const randWalker = function(){

    function clamp(v, min, max){
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    let dir = 0;
    let prevStep = 0;
    const rnd = require('random');

    return { 
        randomWalk(){
            dir = clamp(rnd.normal(dir, 1)(), -0.75, 0.75);
            prevStep += dir;
            return prevStep;
        }
    }
}();

setInterval(()=> {
    // very basic produce, no key, no partitioner, just straight round robin
    target.produce( randWalker.randomWalk().toFixed(4) );
}, 1000);
```

A simple consumer can be implemented as follows:

```javascript
const KafkaRest = require('kafka-rest');

const kafka = new KafkaRest({ 'url': 'http://localhost:8082' });

 // start reading from the beginning
let consumerConfig = {
    'auto.offset.reset' : 'smallest'
};

// join a consumer group on the matches topic
kafka.consumer("consumer-group").join(consumerConfig, function(err, instance) {
      if (err) 
        return console.log("Failed to create instance in consumer group: " + err);

      console.log("Consumer instance initialized: " + instance.toString());
      const stream = instance.subscribe("random_walk");

      stream.on('data', function(msgs) {

          for(var i = 0; i < msgs.length; i++) {
              let key = msgs[i].key.toString('utf8');
              let value = msgs[i].value.toString('utf8');
              console.log(`${key} : ${value}`);

          }

      });

  });
```

### Apache Kafka Consumer Groups

Consumer Groups provide the core abstration on which the proposed architecture is built. Consumer Groups allow a set of concurrent processes to consume messages from a Kafka topic while, at the same time, guaranteeing that no two consumers will be allocated the same list of partitions. This allows seamless scaling up and down of the consumer group as the result of fluctiating traffic, as well as due to restarts caused by consumer crashes. Consumers from a consumer group receive a `rebalance` notification callback with a list of partitions that are allocated to them, and they can resume consuming either from the beginning of the paritition, from the last committed offest by another member of the group or from a self-managed offset.

The code above allows a very easy way to check how consumer groups work. Just start several competing consumers and kill one of them or restart it later. Since restart policy is set to the beginning of the topic, `'auto.offset.reset' : 'smallest'` consuming will start everytime from the beginning of each partition.

### Architecture Description

![Architecture]({{site.url}}/assets/arch_1.png)



