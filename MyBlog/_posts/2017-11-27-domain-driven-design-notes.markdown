---
layout: post
title:  "Domain Driven Design Notes"
date:   2017-11-27 13:15:16 +0200
categories: architecture
---
A very high level introduction to Domain Driven Design (DDD). DDD is a useful technique if the application we are building has complex business functionality. It will not help with big data, performance or hardware specific optimisations. 

### Notes

The content in this article is based on several Pluralsight classes from the ASP.NET track. The examples I wrote in Java The post is infused with my interpretation and comments, so I recommend going directly to the source for the original content:

- [Domain-Driven Design Fundamentals](https://app.pluralsight.com/library/courses/domain-driven-design-fundamentals/table-of-contents)

- [Domain-Driven Design in Practice](https://app.pluralsight.com/library/courses/domain-driven-design-in-practice/table-of-contents)

- [Clean Architecture, Patterns And Principles](https://app.pluralsight.com/library/courses/clean-architecture-patterns-practices-principles/table-of-contents)

- [Modern Software Architecture: Domain Models, CQRS, and Event Sourcing](https://app.pluralsight.com/library/courses/modern-software-architecture-domain-models-cqrs-event-sourcing/table-of-contents)

- [UX-driven Software Design](https://www.pluralsight.com/courses/ux-driven-software-design)

Probably the most important thing to remember when starting a new a project is not to try to design it all upfront. A comprehensive domain model will most likely prove either expensive to develop or useless and rigid in face of changing business requirements. Even more, as developers usually develop what they understand from the domain and since business is usually not able to validate in depth the abstractions of the model, a comprehensive design up-front is usually proven wrong after a few iterations. 

Dino Esposito has a great slogan, "Stop Modeling, Start Mirroring", meaning that instead of trying to build a comprehensive model of the whole domain, start mirroring the most important use cases and develop from there. 

Before proceeding forward is worth noting that every action in a business system is either a command (which alters the state of the system) or a query (which does not alter the state of the system). This leads to the obvious question: why not separate the two? This pattern is called CQRS ([command-query responsibily segregation](https://martinfowler.com/bliki/CQRS.html)). If we take the CQRS course, we basically split the domain model in two distinct parts:
- The command model - rich in business functionality
- The read model - mostly DTOs, suitable for presentation
As Martin Flowler notes in his wiki, CQRS complicates the  architecture and code and its price may not pay off for a large range of applications. This article is not focused on CQRS, although it might touch the topic from time to time.

### Glossary

- *Problem Domain:* the specific problem the software is trying to solve

- *Core Domain:* the specific differentiator for the business, the thing that makes the business unique and cannot be outsourced. If restricting the definition to the software, it is the part that cannot be delegated to another team.

- *Sub-domain:* separate applications or features the software has to deal with

- *Bounded Context:* a specific responsibility, with clear boundaries, that separate it from the rest of the system

- *Context Mapping:* the process of identifying bounded contexts and their relationship to each other

- *Shared Kernel:* a part of the system that is commonly used by several bounded context (e.g. authentication, that various teams working on various bounded contexts agree to change only upon mutual agreement

- *Ubiquitous Language:* a common language, with very precise terms, that the business and the technical agree to use together. Ubiquitous language follows the meaning and the wording from the business and the same wording can have different meanings in different bounded contexts. The ubiquitous language is used everywhere: conversation, class names, method names and it is not be replaced even with synonyms.

- *Anemic Domain Model:* model with classes focused on state management, not with too much functionality. Usually maps directly to storage and is useful for CRUD operations. For DDD it is an antiparttern because then the logic is delegated to domain services and this creates an artificial decouping which breaks encapsulation and generates more code, brittle, with harder to enforce invariants.

- *Rich Domain Model:* model focused on business rules and behaviors, prefered for DDD

- *Entity:* a mutable class with identity, used for tracking and persistence

- *Value Objects:* an immutable class whose identity is given by the combinations its members (e.g. TimeRange)In DDD it is preferred to try to model first in terms of Value Objects and then switch to Entities. 

- *Services:* a collection of behaviours (functionality) that does not belown elsewhere in the domain

- *Aggregate:* a cluster of associated objects that we treat as a unit for purposes of data changes (e.g. a Product which is assembled from a set of Parts). In DDD, we forbit direct access to parts of the aggregate and we only allow access through the *Aggregate Root*. A good case for one specific class to be the aggregate root is to check if on delete it generates a cascade. If so, it might be a good candidate for being the aggregate root of the cascaded object tree. Another critical criteria for choosing the right *Aggregate Root* is for it to naturally enforce invariants. In other words, the *Aggregate* is a transactional graph of objects while the *Aggregate Root* is the entry point to the graph, with the role of assuring its integrit.

- *Invariant:* a condition that must be true for the system to be in a valid / consistent state

- *Repository:* persistent storage pattern. Only the aggregate roots should be available for querying directly from the repository. Repositories are a collection of objects of a certain type, with more ellaborate querying capabilities. For simple CRUD repositories, one can create a generic implementation for the pattern but, for more complex business rules, it is recommended to have a dedicated repository for each *Aggregate Root*

- *Domain Event:* a class that encapsulates the occurence of an event in a domain model, a good pattern for keeping the classes in a domain open for extension and closed for modification. A preferred naming convetinon is *"Entity_PastAction_Event"* (e.g. "UserLoggedInEvent")

- *Anticorruption Layer:* code that keeps the bounded context insulated from other bounded contexts or external applications or services

### Implementation

Steps:

 - Start with the core domain.
 - Don't introduce several bounded contexts upfront. Start with only one.
 - Look for hidden abstractions; refactor.
 - Prefer value objects to entities. Try to put as much as possible of the business logic in value objects. 
 - Extend to other bounded contexts.

 ### Testing the domain

 When experimenting with the model, it might be feasible not to write tests at first. A lot of refactoring goes on in the beginning and TDD would only make refarctoring harder. However, once the code becomes pretty stable and we ar happy with our abstractions, we can add remaining tests and switch back to TDD. 

 ### Entities and Value Objects

*Entities* have identifier identity, meaning that two entities with the same identifier are equal. IDs are mandatory for entities. 
 
*Value objects* have structural identity. Equality means that all their fields have equal values. They don't have IDs. 
 
It makes sense to have a base class for entities and a base class for value objects. 

Because the Value Objects do not have a lifetime different from that of the Entity that holds them (and they don't have an ID), they should be serialized in the same table as the Entity itself. A good analogy is with an integer; we would not create a separate table to store integers. 

The example below is from a hypothetical DDD snack machine application implementation. Please note the type of code each class contains. Low level algorthms in Value Objects and higher level orchestration in the Entity. Entities can be uniquely identified and restored from the database. A lot of logic is stored inside Value Objects like, for instance, the management of banknotes and coins inside the machine.

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

A simple type, a value object, immutable. It is a part of the intrinsic types of the system, so it does not make sense to persit it in its own table:

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

Here is another value object. It does not have life of its own outside the snack machine, it has a fixed structure, and can be expressed as a set of immutable operations - see the discussion below. Lots of business functionality is stored within the value object itself.

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

At first look, the `MoneyCollection` object does not seem at all immutable, which violates the core characteristic of a value object. However, if you look at its methods they can be expressed as:

```
MoneyCollection a, b, c;
int d;

a = a + b; // MoneyCollection::transferFrom
a = a + d; // MoneyCollection::addMoney
a = a - d; // MoneyCollection::substract
```

In such cases, a decision must be taken: do we refactor to keep the value object immutable, in this case paying the price of a small array allocation and copy, or do we hide the dirty details behind a clean facade? Beside the (small) perfomance hit, here is also visible a certain mismatch between a theoretical concept and the comfort with which it can be expressed in our programming language.

*ID generation and serialization*

If we are to take a purist approach, the IDs should be independent of the database or, at least, not come from an autoincrement column. IDs can be GUIDs (like in the example above) or come from other ID generation service that is independent of the storage layer. The problem with using autoincrement columns is twofold: once it allows the repository abstraction to leak in and, second, it violates the Unit Of Work pattern, which states that all changes should be committed to the database in one single short transaction. 

A clean design does not allow ORM-specific code to pollute the domain objects. This might create additional work for mapping between the DDD entity fields and the persisted entity fields (attention, wording overload), but the hope is that this separation will keep the architecture clean and independent of various underlying technologies. This would allow, for instance, to have a dedicated team working on the business logic, specialized in business processes, and another one on persistence, with specialists in optimized caching, databases and various ORMs. 

### Aggregates

Aggregates are a design pattern that helps us simplify the domain model by gathering several entities under a single abstraction:
- Classes outside the aggregate cannot access the entities within the aggegate except through its root.
- An aggregate is a conhesive whole. It represents a cohesive notion of the domain model.
- Every aggregate has a set of invariants that it maintains during its lifetime. These invariants guarantee the aggregate is always in a consistent state.
- Application services retreive aggregates from the database, perform actions on them and then persist them to the database as a transaction. 
- The invariants that spread across several aggregates should not be expected to be consistent all the time, but rather become eventually consistent.

In our example, let's consider that each `SnackMachine` has a set of slots that are filled with products. The user selects the slot (1-10, let's say) and the product that is topmost is returned. This introduces two new abstractions: `Slot` and `Product`. Unlike `MoneyCollection`, each machine can have a different number of slots and in each slot several products:

```java
class Product{
}

class Slot {
    List<Product> productList;
}

class SnackMachine extends Entity<UUID>{
    List<Slot> slots;
}
```

This setup leads pretty quickly to the idea that while the slot is an integral part of the `SnackMachine`, it requires its own separate serialization to the database. However, a slot by itself does cannot exist in vacuum, so when a snack machine is deleted its associated slots must be deleted as well. Just the same, each `SnackMachine` knows well how many slots it has, so it can enforce this invariant. For implementation, it is good to have a marker abstract class, `AggregateRoot`, which inherits from `Entity` and helps clarify the aggregate boundaries. Thus, the declarations above become:

```java
abstract class AggregateRoot<ID> extends Entity<ID> {} 

class Slot extends Entity<Long> {} 

class SnackMachine extends AggregateRoot<UUID> {}
```

The `Product` abstraction has a life of its own, outside the `SnackMachine`. Therefore, it cannot be part of the `SnackMachine` aggregate, so it makes sense to have it as aggregate root by itself.

```java
class Product extends AggregateRoot<UUID> {} 
```

### Repositories

The idea behind the Repository pattern is to encapsulate the persistence and allow the client code to access the data as if it were stored in memory. The general rule is that it should be a single repository per each aggregate, which makes sense given the fact all entities within an aggregate should be accessible only through the aggregate root.

In its simplest form, a repository base class has only two responsibilities:
- Get an aggregate root by its ID
- Commit the aggregate root changes to the database in a single transaction together, with all its dependent entities.

Although clean and simple, keeping the repository too basic will lead to highly inefficient database access (either too many queries or too much data is transferred from the database). Thus, in most practical cases, repositories contain dedicated methods for queries performed against the database, so that just the right amount of data is fetched.

Do not return partially initialized entities from the repository, because this would break the invariants and this is a great source of bugs. A better approach, for methods that do not require full data, is to create separate DTOs. An example below:

```java
public abstract class Repository<K, T extends AggregateRoot<K>> {
    public abstract T getByID(K key);
    public abstract void save(T obj) throws PersistenceException;
}

public abstract class AggregateRoot<ID> extends Entity<ID> {}

public class MoneyInSnackMachineDTO {
    MoneyCollection mc;
}

public class SnackMachineRepository extends Repository<UUID, SnackMachine>{

    // returns fully initialized SnackMachine aggregate root instances
    public List<SnackMachine> getEmptySnackMachines() {} 

    // return only the money data
    public List<MoneyInSnackMachineDTO> getMoneyInEmptySnackMachines() {} 

    // return only the IDs
    public List<UUID> getEmptySnackMachinesIDs() {} 

}
```

In a CQRS application, there would be two repositories. One for handling commands (changes to the state of the system) and one for handling reads (read-only). These two repositories might reference two different undelying databases, each modelled so that it does its job best (updates vs reads).

Below you can see an example of a splitted repository:

```java
public class SnackMachineCommandsRepository extends Repository<UUID, SnackMachine>{

    // returns read / write entities, suitable for handing commands
    public List<SnackMachine> getEmptySnackMachines() {} 

}

public class SnackMachineReadRepository {

   // Returns read-only DTOs.
   // These could be used directly as view models
    public List<MoneyInSnackMachineDTO> getMoneyInEmptySnackMachines() {} 
}
```

### Bounded Contexts

Bounded contexts can be viewed as namespaces for the ubiquitous language. The same term will usually have different meanings in two separate bounded contexts. They touch all layers of the onion architecture as their terminology will be reflected in UI, database model, domain model entities and application services. The relationship between bounded contexts is explicited in the context map.

Subdomains and bounded contexts are best related to each other as 1:1. There is a distinction though; the subdomain is part of the problem space, while the bounded context is part of the solution space. The 1:1 requirement is not always possible. Imagine a situation in which we have the "Sales" subdomain composed of a legacy application and we need to add a new functionality to it in terms of a totally new module. In this case we have a single subdomain but can opt to structure our solution as two bounded contexts separated by an anticorruption layer. Here are some rules of thumb for separating the bounded contexts, independent of the subdomain:
- Team size: 6 - 8 developers per bounded context max
- Code size: code should "fit your head"; one developer shouldn't have any trouble understanding it all
- There shouldn't be a situation where two teams work on a single bounded context

It is recommended that bounded contexts have:
- separate namespaces (or separate deployment units or processes)
- separate database schemas or separated databases

The greater the isolation is, the easier it is to maintain proper bounderies. However, higher isolation comes at the price of higer development cost and solution complexity.

### Shared Kernel

Assuming that we want to introduce a new bounded context, the `ATM`, to our application, it is obvious that it shares a set of common core objects with our `SnackMachine`, while the rest of the business logic is completely different. The common core, the `Money` and the `MoneyCollection` objects, can be extracted as a shared kernel in a separate component. This, however, implies that the new component will affect two different domains, so changes to it must be very well controlled. Extracting the shared kernel also implies some refactoring as, if we look in the `MoneyCollection` code above, some business functionality from the `SnackMachine` leaks into it. Therefore, we need to make `MoneyCollection` business-agnostic, a true value object.

When deciding what to put in a shared kernel, keep in mind the following:
- Business logic should not be reused in most cases as it alwasy evolves on separate trajectories. E.g. the sales perspective of what a product is is different from the perspective of customer support.
- Infrastructure / utility code could be reused, but should be duplicated if the changes from a team impact negatively the performance of another team.

Overall, try to avoid reusing code between bounded contexts as much as possible.

### Domain Events

A domain event is an event that is significant for the domain. Its job is to:
- Decouple bounded contexts
- Facilitates communication between bounded contexts
- Can also be used for collaboration between entities within a single bounded context

Naming suggestion: as specific as possble: Subject (who) + Past Tense Verb (the action that occured) + Event. For instance `MoneyDepositedEvent`. 

Try to include as little information as possible in an event, preferably serializable and immutable, and independent of the domain model classes. This is because events may, at one point, pass the process boundaries when the application is distributed and because we don't want the order in which the event is handled to matter. We don't want to give the receivers the option to mutate the event. A good guideline is to only use primitive types when declaring a domain event.

A bad example:

```java
public class Person extends Entity<long>{
    long ID;
    String name;
    String surname;
    Date birthdate;
}

class PersonChangedEvent { // Not specific: what exactly has changed?
    Person person; // Introduces a dependecy on this bounded context. 
}
```

A better example:

```java
class Person extends Entity<long> {} 

// specific
// passes only the relevat information
// immutable in the sense that one cannot easily alter the person if receives this event
class BirthdayChangedEvent { 
    private long ID; 
    private Date oldBirthday; 
    private Data newBirthday;

    long getPersonID() { return ID; }
    Date getOldBirthday() { return oldBirthday; }
    Date getNewBirthday() { return newBirthday; }
}
```

Including the `ID` is debatable: can the `Person` be reteived by its ID by the downstream consumer? Is this event intended for the local bounded context or for consumption in another bounded context? If the answer is the latter, we may want to include different identification information in the event.

### Event Transport

Events can be transported through in-memory data structures or through service busses, if the destination is outside the boundaries of the current process. No matter what the transport is, it is good to think of  events as a mechanism for decoupling components and thus always ask yourself the question: what if the transport changes?

Implementation, a simple, in memory, transport:

```java
// marker interface
public abstract class DomainEvent{}

// event handler
public interface IHandler<T extends DomainEvent>{
    public void handleEvent(T event);
}

// a generic domain events class which is the global dispatcher
// or all domain events
// do not use in production -> not properly tested
public class DomainEvents {

    private static 
    ConcurrentHashMap<Class<?>,LinkedList<WeakReference<?>>> eventHandlers = new ConcurrentHashMap<> ();

    public static <U extends DomainEvent, T extends IHandler<U>> 
    void  registerWeakReference(T instance, Class<U> eventType){

        // Holding the weak reference is a design decision.
        // While not convenient from a code perspective (cannot reliably hold lambdas),
        // I am not excited to keep in a global object strong references to anything,
        // to avoid memory leaks.
        // A better approach is not to make this object static for long running applications and
        // just keep it as a per-session or per request variable.

        WeakReference<T> ref = new WeakReference<> (instance);

        LinkedList<WeakReference<?>> lst = eventHandlers.getOrDefault (eventType, new LinkedList<> ());

        synchronized (lst) {
            lst.add (ref);
        }

        eventHandlers.put (eventType, lst);
    }

    public static <U extends DomainEvent> void raiseEvent(U event){

        LinkedList<WeakReference<?>> lst = eventHandlers.getOrDefault (
            event.getClass (), 
            new LinkedList<> ());

        List<?> refList = null;

        // so that we don't hold the lock while all handlers execute
        synchronized (lst){
            refList = lst.stream ()
                    .map (wr-> wr.get ())
                    .filter (r -> r != null)
                    .collect (Collectors.toList());

            lst.removeIf ( wr -> wr.get () == null ); // clean up a little bit
        }

        refList.forEach (t -> ((IHandler<U>)t).handleEvent (event));
    }
}
```

And, of course, an example usage:

```java
import domain.DomainEvent;
import domain.DomainEvents;
import domain.IHandler;

/**
 * Created by alexandrugris on 11/29/17.
 */
public class DDDInJavaMain {

    static class PersonNameChangedEvent extends DomainEvent {}
    static class PersonBirthdayChangedEvent extends DomainEvent {}

    static class PersonNameChangedEventHandler implements IHandler<PersonNameChangedEvent>{

        public PersonNameChangedEventHandler(){
            DomainEvents.registerWeakReference (this, PersonNameChangedEvent.class);
        }

        @Override
        public void handleEvent(PersonNameChangedEvent event) {
            System.out.println ("Event handled");
        }
    }

    public static void main(String... args){

        PersonNameChangedEventHandler handler = new PersonNameChangedEventHandler ();
        DomainEvents.raiseEvent(new PersonNameChangedEvent ());

    }
}
```

### The Unit Of Work Problem

This simple pattern for sending and handling events when they occur has a *major architectural issue* - it breaks the Unit Of Work pattern. For instance, what happens when we need to do a rollback before the Unit Of Work transaction is committed? Or, for some reason, the commit fails? As events have aready been submitted, it is impossible to roll them back as their processing causes ripples inside our Bounded Domain but also inside others, causing the system as a whole to enter an invalid state. 

A better way exits to preserve intact the Unit Of Work: split the event pipeline in *creation* (when the event occurs) and *dispatch* (after the UoW is committed successfully to the database). In the meantime, hold the events in an temporary list, on a per-unit-of-work instance or, if all commits are done through the aggregate roots, as it should be, on a per-aggregate root instance. If we take the aggregate root path, a good place to store the logic for queuing the events and dispatching them when the save transaction succeeds is the `AggregateRoot` base class.

## Events vs Commands

- *Commands*: something you want do happen but might not happen.
- *Events:* something that already happend. 

Commands are best named with present tense, imperative form, like `AddNewBookToLibraryCmd` or `ExtractMoneyCmd`. A command might not complete successfully. You don't need to store commands, although it is a good practice to log them for the purpose of debugging. 

In case Event Sourcing is implemented, events are the single source of truth and they must be persisted and stored virtually forever. Not the commands, but the events, because it is the events that build up to the current state of the system.

### DDD Notes

Prefer the "always valid" approach, that is to always maintain entities in a consistent state. This is preferable to the opposite approach where one checks the validity of the model just before serialization. 

Do not skip validations; perform them at all boundaries.

Prefer the static factory method pattern (or, even better, dedicated factories) to constructors. Creating entities can be a heavyweight operation and exceptions might be thrown. Construction of an entity often has little to do with exploiting it afterwards, so separating construction from the entity itself usualy is a good application of the Single Responsibility Principle. Factories should create whole aggregates, not just specific entities. On the other hand, factories create complexity and if the creation logic is simple, one can postpone the creation of an additional factory until it is actually needed.

Keep domain services stateless.

Do not add domain logic to application services. Application services are designed for orchestration with the outside world.

Domain Driven Design is an approach, not a mechanical recipe. It helps the architect understand the model and provide guidance on how to structure the code, but blind application will only lead to unnecessary complexity. Good rules of thumb are YAGNI and KISS, just like for any other software development approach.

### UX Driven Design

UXDD is a process for software design which starts from the user screens to build the model. The assumption behind it is that users are more likely to give effective feedback to the developers if presented with a concrete visual model of how they will interact with the application. 

Instead of long written documents, the main tools for collaboration are the wireframes which depict the User Experience.

Screens depict all the data that goes in and out of the system for each usecase. Therefore, the first important task for the architect is to iterate with the users until the screens are just right. Not talking about graphics, but about workflows and usability.

By putting the requirements in the concrete form of an (interactive) user interface, UXDD mitigates the following problems:
- Requirements are usually mostly guesses
- Communication is slow and painful
- Because of the two above, lots of development is done by assumption



