---
layout: post
title:  "Big Data Online Data Analytics"
date:   2018-10-15 13:15:16 +0200
categories: programming
---
An example of a classical big data online analytics pipeline composed of Kafka, Cassandra and Elastic Search. The implementation provided is in NodeJS.

### Naive Stream Analytics

The first implementation does as follows:

- Client connects to a data producer (in our case one that generates a random walk);
- Client computes the average and standard deviation of a tumbling window of 2 seconds;
- Client declares an anomaly any value that is outside 4 sigma from the previous window average;

*Data producer:*

Data is produced at a rate of 1 data point per 10 milliseconds and it is broadcasted through websockets to any connected clients. When the last client disconnects, producing is stopped so my laptop does not unnecessary and excessively heat up.

```javascript
let app = require('http').createServer( (req, resp) => {});
let io = require('socket.io')(app);
let rnd = require('random');

let cron = null;
let connectedClients = 0;

let prevStep = 0;
let dir = 0;

function clamp(v, min, max){
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

function randomWalk(){

    dir = clamp(rnd.normal(dir, 1)(), -0.75, 0.75);
    prevStep += dir;

    io.emit('walk', prevStep);
}

io.on('connect', (client) => {
    console.log("Client connected");
    if(cron == null){
        cron = setInterval(randomWalk, 10); // flood this thing!
        connectedClients ++;
    }

});

io.on('disconnect', (client) => {

    connectedClients --;
    if(connectedClients <= 0 && cron != null){
        clearInterval(cron);
        cron = null;
    }

})

app.listen(8080);
```

*The client*

```javascript
let socket = require('socket.io-client')('http://localhost:8080');

let processor = function(){

    let values = [];
    let sigma   = null;
    let avg = null;

    return {
        append : function(v){
            values.push(v);
        },

        clear : function(){
            values = [];
        },

        processAndReset : function(){
            avg = 0;
            values.forEach(v => {
                avg += v / values.length; // avoid overflow
            });
            values.forEach(v => {
                sigma += (v - avg) * (v - avg) / values.length;
            });
            sigma = Math.sqrt(sigma);
            values = [];
            return avg;
        },

        isAnomaly : function(v){
            if(sigma == null)
                return false;
            return Math.abs(v - avg) > 4 * sigma;
        }
    }
}();

socket.on('connect', () => {
    console.log("Connected");

    processor.timer = setInterval(() => {
        let avg = processor.processAndReset();
        console.log(`Moving average: ${avg}`);
    }, 2000);

    processor.clear();
});

socket.on('walk', (value) => {

    if(processor.isAnomaly(value)){
        console.log(`Anomaly detected. ${value}`);
    }
    processor.append(value);
});

socket.on('disconnect', () => {
    console.log("Disconnected");

    if (processor.hasOwnProperty('timer') && processor.timer !== null){
        clearInterval(processor.timer);
        delete processor.timer;
    }
});
```

Obviously, this default implementation has faults:

- If the consumer dies or a network partition occurs, data is lost;
- There is no easy way to distribute data processors to multiple machines. There is no communication between clients and the network topology is fixed;
- Data is not stored anywhere;
- In case the consumer or the producer lags behind, data will queue in memory, eventually leading to a crash and, again, data loss;

### Stream Analytics Advanced

A general topology for stream analytics follows roughly the following components:

- Distributed data collectors that publish the incoming messages to a persistent queue;
- Persistent, highly available queue;
- Distributed processors which might aggregate data from a set of multiple streams;
- Online data storage - in all cases I can think of, decisions taken by the processors take into account an existing data model (as in our case above, the decision on whether the new point is an anomaly or not takes into account the average and the standard deviation of the previous window);
- Insights database - a queriable database used for exploring data in an offline manner, to derive new models and insights.

In our demo we will use Kafka as the message queue, Apache Storm as the distributed processing technology, Cassandra as the online data storage and ElasticSearch as the insights database.

### Apache Kafka

Apache Kafka allows to store the incoming data stream and computation results used for later stages in the pipeline in a fault tolerant, distributed way. Apache Kafka also allows brining new servers to the system in case of high data load.

- In case of replication, each partition will have one server as the leader and configurable other as followers. Leader is managing read / write requests for a specific partition, while followers are managing replication of the leader.
- In Kafka, leadership is defined per partition. That means that a server can be leader for a partition but a follower on another.
- Zookeeper stores consumer offsets per topics.

We will start now with Kafka setup. For this, we are going to use the following docker compose file:

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

Sometimes Kafka service fails to start and it must me restarted with `docker-compose scale kafka=1`. Once the service is up, we can test the configuration as follows:

```bash
docker run --net=oda --rm confluentinc/cp-kafka bash -c "seq 42 | kafka-console-producer --request-required-acks 1 --broker-list kafka:9092 --topic foo && echo 'Produced 42 messages.'"

docker run --net=oda --rm confluentinc/cp-kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic foo --from-beginning --max-messages 42
```

To test the Kafka cluster I created the following script:

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