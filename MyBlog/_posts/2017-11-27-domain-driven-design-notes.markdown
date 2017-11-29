---
layout: post
title:  "Domain Driven Design Notes"
date:   2017-11-27 13:15:16 +0200
categories: architecture
---
A very high level introduction to Domain Driven Design (DDD). DDD is a useful technique if the application we are building has complex business functionality. It will not help with Big Data, performance or hardware specific optimisations. 

### Glossary

- *Problem Domain:* the specific problem the software is trying to solve

- *Core Domain:* the specific differentiator for the business, the thing that makes the business unique and cannot be outsourced. If restricting the definition to the software, it is the part that cannot be delegated to another team.

- *Sub-domain:* separate applications or features the software has to deal with

- *Bounded Context:* a specific responsibility, with clear boundaries, that separate it from the rest of the system

- *Context Mapping:* the process of identifying bounded contexts and their relationship to each other

- *Shared Kernel:* a part of the system that is commonly used by several bounded context (e.g. authentication), that various teams working on various bounded contexts agree to change only upon mutual agreement

- *Ubiquitous Language:* a common language, with very precise terms, that the business and the technical agree to use together. Ubiquitous language follows the meaning and the wording from the business and the same wording can have different meanings in different bounded contexts. The ubiquitous language is used everywhere: conversation, class names, method names and it is not be replaced even with synonyms.

- *Anemic Domain Model:* model with classes focused on state management, not with too much functionality. Usually maps directly to storage and is useful for CRUD operations.

- *Rich Domain Model:* model focused on business rules and behaviors, prefered for DDD

- *Entity:* a mutable class with identity, used for tracking and persistence

- *Value Objects:* an immutable class whose identity is given by the combinations its members (e.g. TimeRange)In DDD it is preferred to try to model first in terms of Value Objects and then switch to Entities. 

- *Services:* a collection of behaviours (functionality) that does not belown elsewhere in the domain

- *Aggregate:* a cluster of associated objects that we treat as a unit for purposes of data changes (e.g. a Product which is assembled from a set of Parts). In DDD, we forbit direct access to parts of the aggregate and we only allow access through the *Aggregate Root*. A good case for one specific class to be the aggregate root is to check if on delete it generates a cascade. If so, it might be a good candidate for being the aggregate root of the cascaded object tree. Another critical criteria for choosing the right *Aggregate Root* is for it to enforce invariants. 

In other words:

- *Aggregate:* a transactional graph of objects

- *Aggregate Root:* the entry point to the graph that assures the integrity of the entire graph

- *Invariant:* a condition that must be true for the system to be in a valid / consistent state

- *Repositories:* persistent storage pattern. Only the aggregate roots should be available for querying directly from the repository. Repositories are a collection of objects of a certain type, with more ellaborate querying capabilities. For simple CRUD repositories, one can create a generic implementation for the pattern, but for more complex business rules, it is recommended to have a dedicated repository for each *Aggregate Root*

- *Domain Event:* a class that encapsulates the occurence of an event in a domain model, a good pattern for keeping the classes in a domain open for extension and closed for modification. Usually they can be expressed as *"Entity_PastAction"* (e.g. "User_LoggedIn")

- *Anticorruption Layer:* code that keeps the bounded context insulated from other bounded contexts or external applications or services

### Implementation

 - Start with the core domain
 - Don't introduce several bounded contexts upfront
 - Look for hidden abstractions; refactor
 - Prefer value objects to entities. Try to put most of the business logic in value objects. Entities act like wrappers upon them.

 Entities have identifier identity, meaning that two entities with the same identifier are equal. IDs are mandatory for entities. Value objects have structural identity. Equality means that all their fields have equal values. They don't have IDs. It makes sense to have a base class for entities and a base class for value objects. 

### Testing the domain

 When experimenting with the model, it might be feasible not to write tests at first. A lot of refactoring goes on in the beginning and TDD would only make refarctoring harder. However, once the code becomes pretty stable and we ar happy with our abstractions, we can add remaining tests and switch back to TDD. 

### Back To Implementation / Serialization

Because the Value Objects do not have a lifetime different from that of the Entity that holds them (and they don't have an ID), they should be serialized in the same table as the Entity itself. A good analogy is with an integer; we would not create a separate table to store integers. 

Examples below are from a DDD SnackMachine application implementation. Please look at the code and note the type of code each class contains. Low level algorthms in Value Objects and higher level orchestration in the Entity.

*Entities and Value Objects*

This is an Entity. It can be uniquely identified and restored from the database. It has business functionality but, in most of the cases, orchestrates the value objects inside it. A lot of logic is stored inside Value Objects like, for instance, the management of banknotes and coins inside the machine.

```java
public class SnackMachine extends Entity<UUID> {

    // these properties need to be serialized,
    // so in case of a crash the system restores to its normal function.
    private MoneyCollection moneyInDeposit = new MoneyCollection();
    private MoneyCollection moneyInTransaction = null;

    // business functionality
    public void insertMoney(Money am, int count){

        moneyInTransaction = (moneyInTransaction == null)? new MoneyCollection() : moneyInTransaction;
        moneyInTransaction.addMoney(am, count);
    }

    // business functionality
    public MoneyCollection returnMoney(){

        MoneyCollection returned = moneyInTransaction.getChange ();
        moneyInDeposit.transferFrom(moneyInTransaction);
        moneyInTransaction = null;

        return returned;
    }

    // business functionality
    public void buySnack(int costInCents) throws NotEnoughMoneyException{

        if(moneyInTransaction == null || moneyInTransaction.getTotalMoney() < costInCents)
            throw new NotEnoughMoneyException();

        moneyInTransaction.substract(costInCents);

        // TODO: return the snack and persist the transaction
    }

    public int remainingMoneyInTransaction(){
        return (moneyInTransaction != null)? moneyInTransaction.getTotalMoney () : 0;
    }

    public MoneyCollection getMoneyInDeposit(){
        return moneyInDeposit;
    }

    @Override
    protected UUID generateID(){
        return UUID.randomUUID();
    }
}
```

This is a simple type - a value object, immutable. It is a part of the intrinsic types of the system, so it does not make sense to persit it in its own table. 

```java
public enum Money{

    TEN_CENTS   (10),
    FIFTY_CENTS (50),
    ONE_DOLLAR  (100),
    TEN_DOLLARS (1000),
    TWENTY_DOLLARS (2000),
    FIFTY_DOLLARS  (5000);

    private int value = 0;

    Money(int _value){
        value = _value;
    }

    public int getValue(){ return value; }

    public int getValue(int count){
        return value * count;
    }
}
```

Here is another value object. It does not have life of its own outside the snack machine, it has a fixed structure, and can be expressed as a set of immutable operations - see the discussion below. As seen below, lots of business functionality is stored within the value object itself.

```java
public class MoneyCollection{

    /** A set with the count of each type of coins in the machine */
    private int[] moneyDeposit = new int[Money.values().length];
    private int substractedSum = 0;

    public int getCountOf(Money am){
        return moneyDeposit[am.ordinal()];
    }

    public void addMoney(Money am, int count){
        moneyDeposit[am.ordinal()] += count;
    }

    public void transferFrom(MoneyCollection mc){
        for(int i = 0; i < moneyDeposit.length; i ++){
            moneyDeposit[i] += mc.moneyDeposit[i];
            mc.moneyDeposit[i] = 0;
        }
    }

    public int getTotalMoney(){
        return IntStream.range(0, Money.values().length)
                .map(i -> moneyDeposit[i] * Money.values()[i].getValue())
                .sum() - substractedSum;
    }

    public void substract(int value) throws NotEnoughMoneyException{
        if(value > getTotalMoney()) throw new NotEnoughMoneyException();
        substractedSum += value;
    }

    /**
     * Returns the coins associated with the value.
     */
    public MoneyCollection getChange(){

        MoneyCollection ret = new MoneyCollection();
        Money[] values= Money.values();

        int value = getTotalMoney ();

        // simple greedy return change
        int i = values.length - 1;
        while(value > 0){

            while(i >= 0 && (values[i].getValue() > value || moneyDeposit[i] == 0))
                i --;

            if(i > 0 || (i == 0 && values[i].getValue() <= value && moneyDeposit[i] > 0)){
                ret.addMoney(values[i], 1);
                moneyDeposit[i] --;
                value -= values[i].getValue();
            }
            else
                break; // nothing to give anymore
        }

        substractedSum = value;
        return ret;
    }
}
```

The `MoneyCollection` object does not seam at all immutable, except for its internal array size, which violates the core characteristic of a value object. However, if you look at its methods they can be expressed as:

```
MoneyCollection a, b, c;
int d;

a = a + b; // MoneyCollection::transferFrom
a = a + d; // MoneyCollection::addMoney
a = a - d; // MoneyCollection::substract
```

In such cases, a decision must be taken: do we refactor to keep the value object immutability characteristic, in this case paying the price of a small array allocation and copy, or do we hide the dirty details behind a clean facade? Beside the (small) perfomance hit, here is also visible a certain mismatch between a theoretical concept and the easiness in which it can be expressed in our programming language.

*ID generation and serialization*

If we are to take a purist approach, the IDs should be independent of the database or, at least, not come from an autoincrement column. IDs can be GUIDs (like in the example above) or come from other ID generation service that is independent of the storage layer. The problem with using autoincrement columns is twofold: once it allows the repository abstraction to leak in and, second, it violates the Unit Of Work pattern, which states that all changes should be committed to the database in one single short transaction. 

A clean design does not allow ORM-specific code to pollute the domain objects. This might create additional work for mapping between the DDD entity fields and the persisted entity fields (attention, wording overload), but the hope is that this separation will keep the architecture clean and independent of various underlying technologies. This would allow, for instance, to have a dedicated team working on the business logic, specialized in business processes, and another one on persistence, with specialists in optimized caching, DB queries, various ORMs. 

### Aggregates

A design pattern that helps us simplify the domain model by gathering several entities under a single abstraction:
- Classes outside of the aggregate cannot access the entities within the aggegate except through its root.
- An aggregate is a conhesive whole. It represents a  cohesive notion of the domain model.
- Every aggregate has a set of invariants that it maintains during its lifetime. These invariants guarantee the aggregate is in a consistent state.
- Application services retreive aggregates from the database, perform actions on them and then persist them to the database as a transaction. However, the invariants that spread across several aggregates should not be expected to be consistent all the time, but rather become eventually consistent.

In our example, let's consider that each `SnackMachine` has a set of slots that are filled with products. The user selects the slot (1-10, let's say) and the product that is topmost is returned. This introduces two new abstractions: `Slot` and `Product`. Unlike `MoneyCollection`, each machine can have a different number of slots and in each slot several products:

```java
class Product{
}

class Slot {
    IList<Product> productList;
}

class SnackMachine extends Entity<UUID>{
    IList<Slot> slots;
}
```

This setup leads pretty quickly to the idea that while the slot is an integral part of the `SnackMachine`, it requires its own separate serialization to the database. However, a slot by itself does cannot exist in vacuum, so when a snack machine is deleted its associated slots must be deleted as well. Just the same, each `SnackMachine` knows well how many slots it has, so it can enforce this invariant. For implementation, it is good to have a marker abstract class, `AggregateRoot`, which inherits from `Entity` and helps clarify the aggregate boundaries. Thus, the declarations above become:

```java
abstract class AggregateRoot<ID> extends Entity<ID> {} 

class Slot extends Entity<Long> {} 

class SnackMachine extends AggregateRoot<UUID> {}
```

However, the `Product` abstraction has a life of its own, outside the SnackMachine. Therefore, it cannot be part of the `SnackMachine` aggregate but rather it is an aggregate root by itself.

```java
class Product extends AggregateRoot<UUID> {} 
```








