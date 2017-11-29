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

Examples below are from a DDD SnackMachine application implementation.

*Entities and Value Objects*

This is an Entity. It can be uniquely identified and restored from the database. It has business functionality but, in most of the cases, orchestrates the value objects inside it. A lot of logic is stored inside Value Objects like, for instance, the management of banknotes and coins inside the machine.

```java
public class SnackMachine extends Entity<UUID>{

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
        moneyInDeposit.transferFrom(moneyInTransaction.getSubstractedMoney());
        MoneyCollection returned = moneyInTransaction;
        moneyInTransaction = null;
        return returned;
    }
    // business functionality

    public void buySnack(int costInCents) throws NotEnoughMoneyException{
        if(moneyInTransaction == null || moneyInTransaction.getTotalMoney() < costInCents)
            throw new NotEnoughMoneyException();

        moneyInTransaction.substract(costInCents);
    }

    @Override
    protected UUID generateID(){
        return UUID.randomUUID();
    }
}
```

This is a simple type - a value object. It is a part of the intrinsic types of the system, so it does not make sense to persit it in its own table. 

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

This is another value object. It does not have life of its own outside the snack machine, so it should not be serialized outside the snack machine. As seen below, lots of business functionality is stored within the value object itself.

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
            .sum();
    }

    public void substract(int value) throws NotEnoughMoneyException{
        if(value > getTotalMoney()) throw new NotEnoughMoneyException();
        substractedSum -= value;
    }

    /**
    * Returns the coins associated with the value.
    */
    public MoneyCollection getSubstractedMoney(){

        MoneyCollection ret = new MoneyCollection();

        Money[] values= Money.values();

        // simple greedy return change
        int i = values.length - 1;     
        while(value > 0){

            while(i >= 0 && (values[i].getValue() > substractedSum || moneyDeposit[i] == 0))
                i --; 
            
            if(i > 0 || (values[i].getValue() <= substractedSum && moneyDeposit[i] > 0)){
                ret.addMoney(values[i], 1);
                moneyDeposit[i] --;
                substractedSum -= values[i].getValue();
            }
            else
                return ret; // nothing to give anymore
        }
        
        return ret;
    }
}
```

*ID generation and serialization*

If we are to take a purist approach, the IDs should be independent of the database or, at least, not come from a autoincrement column. They can be GUIDs (like in the example above) or from other ID generation service that is independent of storage. The problem with using autoincrement columns is twofold: once it allows the repository abstraction to leak and, second, it violates the Unit Of Work pattern, which states that all changes should be committed to the database in one single short transaction. 










