---
layout: post
title:  "RabbitMQ Patterns and Considerations"
date:   2017-06-03 13:15:16 +0200
categories: distributed systems
---
A short intro into RabbitMQ and its C# client. The code which merges most of the concepts from this article can be found [here](https://github.com/alexandrugris/RabbitMQ_Play).

### How to run RabbitMQ

By far the easiest and most portable way to run RabbitMQ is to use the official docker container with the management console started: 

```
docker run -d --rm --hostname my-rabbit -p 4369:4369 -p 15671-15672:15671-15672 -p 5672:5672 \
    --name my_rabbit_mq rabbitmq:3-management
```

The corresponding connetion string is `"amqp://guest:guest@localhost:5672"` and the management URL: `http://localhost:15672/#/ `

### Basic concepts

Each service (application) maintains one connection to the queue. Connections are made to be shared across threads.

Within a connection, one or more channels can coexist to provide for concurrency. Rule of thumb: 1 channel / thread. Channels are not meant to be shared across threads. The connection object is. Inside RabbitMQ, each channel is served by an Erlang thread (lightweight actor pattern, Erlang can spawn huge amount of threads).

Producers write to an exchange. Exchanges can communicate with queues or other with exchanges through binding. Consumers read from queues. One application can read from one or more queues. In a configurtion with exactly one producer and one consumer, the oldest message is consumed first. RabbitMQ provides strong guarantees for this.

Only when the queue receives an ACK, the message is deleted from the queue. Producers write to exchanges and attach to each message a routing key. The exchange will route the message to the corresponding queue based on the routing key and the exchange type. Exchanges can be of several types: Direct, Topic, Fanout, Headers. Below is a summary of each exchange type and the associated routing behavior:

*Direct Exchange*

Messages are routed to the specified queue using the routing key.

*Topic Exchange*

Routing behaves very much like for the direct exchange. However, routing keys can have several terms separated by dots. E.g. `package.fast.international`. Queues listen to various keys by using wildcards. E.g. `package.*.international`. `*` is the wildcard for one word. `#` is the hashtag for multiple words.

*Fanout Exchange*

The routing key is ignored. Message is sent to all bound queues.

*Headers Exchange*

Routing is based on the message headers which are set through `IBasicProperties::Headers` property. Matching is done for all headers or for any.

[More details here](https://www.rabbitmq.com/tutorials/amqp-concepts.html)

### Creating a connection and then a channel

```csharp
var cf = new RabbitMQ.Client.ConnectionFactory
{
	Uri = Commons.Parameters.RabbitMQConnectionString,
	[...]
};

conn = cf.CreateConnection(); // one per application
[...]
IModel chan = conn.CreateModel(); // one per thread
```

### Setting up the routing topology

Use:

- `IModel::ExchangeDeclare()` to declare an exchange
- `IModel::QueueDeclare()` to declare a queue
- `IModel::QueueBind()` to bind a queue to an exchange

Beside the normal exchange types, two special exchanges stand out:

- *Alternate Routing Exchange:* useful for routing messages which cannot be routed according to the predefined rules and otherwise would have been dropped.

- *Dead Letter Exchange:*  messages that have been rejected or messages that have their TTL expired are routed here. The dead letter exhange can be used for scheduling messages at a specific time, by setting their TTL property.

Here is an example on how to declare such a topology, with a dead letter exchange (DLX) and alternate routing exchange set to the same exchange instance:

```csharp
 IModel chan = ...;
 [...]
 chan.ExchangeDeclare(
     Commons.Parameters.RabbitMQExchangeName_DLX,
     ExchangeType.Fanout,
     durable: false,
     autoDelete: false,
     arguments: null
 );

 // to simplify the topology,
 // we will use the same dead letter exchange as alternative exchange in case of routing failures
 chan.ExchangeDeclare(
         exchange: Commons.Parameters.RabbitMQExchangeName,
         type: ExchangeType.Direct, // change to Fanout to send to several queues
         durable: false, // no serialization
         autoDelete: false,
         arguments: new Dictionary<string, object>()
         {
             { "alternate-exchange", Commons.Parameters.RabbitMQExchangeName_DLX }
         }
  );
 
 chan.QueueDeclare(
     queue: Commons.Parameters.RabbitMQQueueName,
     durable: false,
     exclusive: false,
     autoDelete: false,
     arguments: new Dictionary<string, object>()
     {
         { "x-dead-letter-exchange", Commons.Parameters.RabbitMQExchangeName_DLX }
     }
 );

 chan.QueueDeclare(
     queue: Commons.Parameters.RabbitMQQueueName_DLX,
     durable: false,
     exclusive: false,
     autoDelete: false,
     arguments: null
 );

 chan.QueueBind(
     queue: Commons.Parameters.RabbitMQQueueName,
     exchange: Commons.Parameters.RabbitMQExchangeName,
     routingKey: "RabbitMQ_Play");

 /**
  * The dead-lettering process adds an array to the header of each dead-lettered message named x - death.
  * This array contains an entry for each dead lettering event, identified by a pair of { queue, reason}. 
  * https://www.rabbitmq.com/dlx.html
  */

 chan.QueueBind(
     queue: Commons.Parameters.RabbitMQQueueName_DLX,
     exchange: Commons.Parameters.RabbitMQExchangeName_DLX,
     routingKey: ""
 );
```

In the code above several parameters have been used to declare exchanges and queues. Here are their meaning:

 - *durable: false* : messages will not be persisted to disk. Even if set to true, each message should have the durable flag turn on for persistence
 - *exclusive: false* : if set to true, messages can only be consumed by this connection. Anyone can publish though. When set to true, this configuration is used in the RPC and scatter-gather usage patterns as reply queues.
 - *autoDelete: false* : if true, the queue is deleted when there are no more consumers. However, if there are no consumers ever on the queue, it is not deleted.

### Sending messages

We may want confirmation that the message has been received by the queue:

```csharp
// for publisher to get confirmation that the message has been received by the queue: 
chan.ConfirmSelect();
chan.BasicAcks += (o, args) =>  Console.WriteLine($"Msg confimed {args.DeliveryTag}"); 
chan.BasicNacks += (o, args) => Console.WriteLine($"Error sending message to queue {args.DeliveryTag");
```

Then set the message properties and headers and call:

```csharp
chan.BasicPublish(Commons.Parameters.RabbitMQExchangeName, routingKey, msgProps, Encoding.UTF8.GetBytes(msg));
```

### Receiving messages

Inside the client, for receiving messages, one can set `prefetchCount` to load multiple messages. However, if the server crashes, these will all remain unacknowledged even if processed.

```csharp
if (cthread != System.Threading.Thread.CurrentThread.ManagedThreadId)
    throw new Exception("Channel reused from a different thread");

chan.BasicQos(
          prefetchSize: 0, // no limit
          prefetchCount: 1, // 1 by 1
          global: false // true == set QoS for the whole connection or false only for this channel
          );
chan.BasicConsume(Commons.Parameters.RabbitMQQueueName, noAck: false, consumer: this);
```

The `consumer: this` in the listing above refers to the `Consumer` class below which extends the `DefaultBasicConsumer` class:

```csharp
class Consumer : DefaultBasicConsumer, IDisposable
{
    private IModel chan = null;    
    private int cthread = System.Threading.Thread.CurrentThread.ManagedThreadId;
   
   [...]
   
    // callback for each received message
    public override void HandleBasicDeliver(string consumerTag, 
        ulong deliveryTag, 
        bool redelivered, 
        string exchange, 
        string routingKey, 
        IBasicProperties properties, 
        byte[] body)
    {
        [...]            
    }
```

Another way to go is to use the `QueuingBasicConsumer(model)` and then `(BasicDeliveryEventArgs)consumer.Queue.Dequeue();` for extracting the message in a loop.

### Reliability options

*Acks* - Rabbitmq only deletes a message from the queue when the message is acknowledged by the consumer. Can be set to off in the consumer, which means the message is deleted as soon as it is delivered. The consumer is notified if a message is redelivered by a `redelivered == true` flag.

*Publisher confirms* - for the publisher to know that a message has been queued or not. In case of important messages, implement a re-send strategy for the cases when the queue is not accessible.

```csharp
chan.ConfirmSelect();
chan.BasicAcks += (o, args) => Console.WriteLine($"Msg confimed {args.DeliveryTag}");
chan.BasicNacks += (o, args) => Console.WriteLine($"Error sending message to queue {args.DeliveryTag}");
```

*Mandatory* - set as a flag in `BasicPublish`. If the message cannot be routed to the queue it will be sent back to the producer. By default, if the flag is not set, the message is lost. The event `BasicReturn` is fired on the channel. Routing failures can be treated using the alternative exchange feature.

*Reply to sender* - producer is notified when the consumer has received the message. Use the `ReplyTo` field in message properties or use `SimpleRpcServer` and `SimpleRpcClient`. [Example here](https://www.rabbitmq.com/tutorials/tutorial-six-dotnet.html)

*Connection and topology recovery* - retry in case of failure to send messages, only if the queues and exchanges are set to durable. Even if the topology is set to durable, the messages are lost if their individual flag for durability is not set.

```csharp
var cf = new RabbitMQ.Client.ConnectionFactory
{
	Uri = Commons.Parameters.RabbitMQConnectionString,
	AutomaticRecoveryEnabled = true,
	TopologyRecoveryEnabled = true,
	NetworkRecoveryInterval = TimeSpan.FromSeconds(5),
	UseBackgroundThreadsForIO = false //Foreground threads keep the app alive until finished
};

conn = cf.CreateConnection();
```

### Supported Routing Scenarios:

*Basic patterns:*

- Simple one-way messaging (Exchange type: direct, message sent to unnamed (default queue))
- Worker queues (Exchange type: direct, several consumer listening to the same queue, reading the messages in a round-robin fashion if all are waiting)
- Publish-subscribe (Exchange type: fan-out, routing key is ignored, message is sent to all queues bound to the exchange)
- RPC (Exchange type: direct, message can be sent to default exchange with a specified routing key and response is received on a specified unique response queue, owned by the client)

*Advanced patterns:*

- Routing (Exchange Type: direct, message is sent to a named exchange, routing key is specified so information only reaches the queues matching the pattern)
- Topic (Exchange type: topic. Routing key is a string separated by dots and wildcards. E.g.: "ro.alexandrugris.*".)
- Headers (Exchange type: headers. Message is sent to the queues which match the headers. Routing key should not be set. Match type should indicate if all or any header must match)
- Scatter-gather (Exchange type: can be any, routing key is optional depending on the exchange type. The sender will start by creating and polling a response queue and then dispatch its request)

These are covered extensively in the [RabbitMQ tutorials](http://www.rabbitmq.com/getstarted.html).

### Dealing with errors

*Scenario 1: exception is caught in the consumer and `chan.BasicNack(resend: true)` is sent to the queue.*

The message is then immediately redispatched to a consumer with the flag `redelivered == true`. However there is no mechanism to know how many retries have occured. Thus, a better alternative is to requeue the message again.

*Scenario 2: exception is caught and the message is redelivered to the queue for a number of times.* 

The message is posted back at the beginning of the queue, so the retry will happen only after all other messages have been consumed. In order to keep track of the number of retries, a header is set in the properties which is decreased with each retry. After resubmitting the message back the the queue, the failed message is ACKed. When the resubmit count reaches 0, the message is rejected. If a dead letter queue is specified in the routing topology, the message is automatically directed by RabbitMQ to this queue. Otherwise it is silently dropped.

A strategy that is mixing both approaches is implemented in the code below:

```csharp
public override void HandleBasicDeliver(string consumerTag, 
        ulong deliveryTag, 
        bool redelivered, 
        string exchange, 
        string routingKey, 
        IBasicProperties properties, 
        byte[] body)
{
    [...]
    if ([MessageProcessedSuccessfully])
    {
        // send ack only for this message and only if no error so far
        chan.BasicAck(deliveryTag, false); 
    }
    else // error condition
    {
        if (redelivered || GetRetryCount(properties) < MAX_RETRY_COUNT)
        {
            Requeue(consumerTag, deliveryTag, exchange, routingKey, properties, body);
        }
        else
        {
            // first time simply put it back in the queue for another try
            chan.BasicNack(deliveryTag, false, true);
        }
    }
}
private int GetRetryCount(IBasicProperties properties)
{
    // use the headers field of the message properties to keep track of 
    // the number of retries
    return (int?)properties.Headers?["Retries"] ?? MAX_RETRY_COUNT;
}
private void SetRetryCount(IBasicProperties properties, int retryCount)
{
    properties.Headers = properties.Headers ?? new Dictionary<string, object>();
    properties.Headers["Retries"] = retryCount;
}
private void Requeue(string consumerTag, 
    ulong deliveryTag, 
    string exchange, 
    string routingKey, 
    IBasicProperties properties, 
    byte[] body)
{
    int retryCount = GetRetryCount(properties);
    Console.WriteLine($"Retry count: {retryCount}");
    if (retryCount > 0)
    {
        SetRetryCount(properties, --retryCount);
        chan.BasicPublish(exchange, routingKey, properties, body);
        // this is slow, but we need to make sure somehow the message reaches the queue back
        // channel is configured with chan.ConfirmSelect()
        chan.WaitForConfirmsOrDie(); 
        chan.BasicAck(deliveryTag, false);
    }
    else
    {
        // reject the message to dead letter queue.
        chan.BasicNack(deliveryTag, false, false); 
    }
}
```

In the code above there is no control over when the retry will occur. This is generally not a good strategy because the system might have not yet recovered from the conditions that led to the error in the first place. A better approach is to set the TTL to the message and then push it to another waiting queue for which there is no consumer, but has attached this queue to its dead letter exchange. When the TTL expires, RabbitMQ automatically moves the message to the dead letter exchange (and from here routed to our queue) from which we can consume it once again.

Patterns to take into consideration:

- Dead letter exchange (DLX): RabbitMQ moves TTL-expired messages and rejected messages to this exchange. DLX can have any type and with various queues attached for custom routing. 

- Routing error: alternative queue - a queue declared for messages that cannot be routed to any other queue and would otherwise be silently dropped

- Scheduled delivery / redelivery: together with the DLX, the messages have a TTL after which they are moved to the DLX

In my demo the producer creates the routing topology with:
- One exchange of type direct (`alexandrugris.1st_exchange`)
- One one alternate exchange which is also the dead letter exchange (`alexandrugris.1st_exchange_dead_letter_exchnage`)

![Exchanges]({{site.url}}/assets/rabbitmq_3.png)

And two queues:
- One queue connected to the the `1st_exchange`
- One connected to the dead letter exchange

The producer dispatches roughly 99% of the messages to the right queue and 1% of the messages have a bogus routing key. Thus, if the producer is run without a consumer, we get the following:

![Producer run without a consuming client]({{site.url}}/assets/rabbitmq_1.png)

It is visible that some messages are sent to the alternate exchange queue due to failed routing.

The producer also sets a TTL to all the messages that are sent. Thus, after roughly a minute, we get the following:

![Some messages moved to the DLX, some messages deleted]({{site.url}}/assets/rabbitmq_2.png)

Two things are worth noting:
- The main bulk of the messages that were not consumed are moved to the dead letter queue
- The messages that were already in the dead letter queue, when their TTL expires, are deleted as there is no other destination for them

Because after the movement the TTL is reset for all messages, the situation remains stable until the server is reset.
 

### Message persistence and serialization

Durability of a queue does not make messages that are routed to that queue automatically durable. If a broker is taken down and then brought back up, a durable queue will be re-declared during broker startup, however, only persistent messages will be recovered. 

```csharp
var props = model.CreateBasicProperties();
props.SetPersistent(true);
```

To set properties for the message, `model.CreateBasicProperties()` returns the following interface instance:

```csharp
public interface IBasicProperties{
   
    string UserId { get; set; }
    /// Message timestamp.
    AmqpTimestamp Timestamp { get; set; }
    // Convenience property; parses RabbitMQ.Client.IBasicProperties.ReplyTo property
    // using RabbitMQ.Client.PublicationAddress.Parse(System.String), /// and serializes
    // it using RabbitMQ.Client.PublicationAddress.ToString.
    PublicationAddress ReplyToAddress { get; set; }
    // Destination to reply to.
    string ReplyTo { get; set; }
    // Message priority, 0 to 9. ///
    byte Priority { get; set; }
    // Sets RabbitMQ.Client.IBasicProperties.DeliveryMode 
    // to either persistent (2) or non-persistent (1). ///
    bool Persistent { get; set; }
    // Application message Id. ///
    string MessageId { get; set; }
    // Message header field table. Is of type System.Collections.Generic.IDictionary`2.        
    IDictionary<string, object> Headers { get; set; }
    // Message expiration specification. ///
    string Expiration { get; set; }
    // Non-persistent (1) or persistent (2). ///
    byte DeliveryMode { get; set; }
    // Application correlation identifier. ///
    string CorrelationId { get; set; }
    // MIME content type. ///
    string ContentType { get; set; }
    // MIME content encoding. ///
    string ContentEncoding { get; set; }
    // Intra-cluster routing identifier (cluster id is deprecated in AMQP 0-9-1).        
    string ClusterId { get; set; }
    // Application Id. ///
    string AppId { get; set; }
    // Message type name. ///
    string Type { get; set; }
}
```

With the associated "Is[PropName]Present()" one can check if the property has been set. 

Here is a usage example:

```csharp
var msgProps = chan.CreateBasicProperties();
msgProps.ContentType = "application/json";
msgProps.CorrelationId = Guid.NewGuid().ToString(); // set a correlation id to the message
chan.BasicPublish(Commons.Parameters.RabbitMQExchangeName, "", msgProps, Encoding.UTF8.GetBytes(msg));
```

*Serialization*

1. Convert object to byte[] array
2. Indicate the message `IBasicProperties::Type` (what kind of object is serialized, for instance `ro.alexandrugris.Person`)
3. Indicate the format of serialization and encoding, using `IBasicProperties::ContentType` and `IBasicProperties::ContentEncoding`
4. Send the message

*Deserialization*

1. Receive the message
2. Select deserializer based on content type and content encoding
3. Workout the message type to select the deserialization target
4. Deserialize the message

Ideally the serialization should be application agnostic (for instance a fully qualified class name might not be a good solution if the message is to be consumed across tehnology boundaries) and, of course, versioning should be taken into account.

For binary messages, a potential encoding could be `Protocol Buffers` or `Apache Thrift` which support versioning as well as multiple language bindings.

AMQP supports very large messages. However, if the application requires sending such messages it is better to use a separate, dedicated instance of the server in order not to impact the throughput of smaller, more frequent exchanges. 

Approaches for sending large messages:

- Buffered (small to medium files) - attention to memory performance.
- Chunked (larger files) - use headers to keep track on how to assemble back the file on the receiving end. Attention to `Worker Queue` pattern where the [chunks will not all end up on the same machine](https://stackoverflow.com/questions/21363302/rabbitmq-message-order-of-delivery).

```csharp
var props = model.CreateBasicProperties();
props.SetPersistent(true);
props.Headers = new ListDictionary();
props.Headers.Add("FileName", fileName);
props.Headers.Add("ChunkNumber", chunkNumber);
props.Headers.Add("EOF", eof);
```
Best though is better to avoid transferring large files.

### Notes

*Federation*

RabbitMQ supports [Federation](https://www.cloudamqp.com/blog/2015-03-24-rabbitmq-federation.html) for scaling out to another cluster or datacenter.

*How many queues?*

Can rabbitmq be used as the infrastructure for a chat server in which each person is modelled as an actor? More precise, how many queues can a rabbitmq instance support?

The answer is yes, as the limit is not in the number of queues but in the number of TCP connections supported on a machine. For many connections it is better to have a rabbitmq cluster. 

 - [How many queues on a single server](https://stackoverflow.com/questions/22989833/rabbitmq-how-many-queues-rabbitmq-can-handle-on-a-single-server)
 - [How many queues on a single server](http://rabbitmq.1065348.n5.nabble.com/How-many-queues-can-one-broker-support-td21539.html)
 - [Distributed RabbitMQ](https://www.rabbitmq.com/distributed.html)

*Correlation ID and Microservices*

As each microservice is persisting its data in its own private database, with private indices, one needs a method for correlating various messages into a single logical entity. RabbitMQ provides a correlation ID property for messanges. A good value for it can be GUID. 

Correlation ID is also used in the RPC pattern so that the caller can track for which request each answer came.

 - [Correlation ID](http://jeftek.com/178/what-is-a-correlation-id-and-why-do-you-need-one/)
 - [Practical examples for correlation IDs](https://stackoverflow.com/questions/20184755/practical-examples-of-how-correlation-id-is-used-in-messaging)
 - [RPC-like calls](http://www.rabbitmq.com/tutorials/tutorial-six-dotnet.html)

