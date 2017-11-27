---
layout: post
title:  "Domain Driven Design Notes"
date:   2017-11-27 13:15:16 +0200
categories: architecture
---
A very high level introduction to Domain Driven Design Terminology.

### Glossary

- *Problem Domain:* the specific problem the software is trying to solve

- *Core Domain:* the specific differentiator for the business, the thing that makes the business unique and cannot be outsourced

- *Sub-domain:* separate applications or features the software has to deal with

- *Bounded Context:* a specific responsibility, with clear boundaries, that separate it from the rest of the software

- *Context Mapping:* the process of identifying bounded contexts and their relationship to each other

- *Shared Kernel:* a part of the system that is commonly used by several bounded context (e.g. authentication), that various teams working on various bounded contexts agree to change only upon mutual agreement

- *Ubiquitous Language:* a common language, with very precise terms, that the business and the technical agree to use together. Ubiquitous language follows the meaning and the wording from the business and the same wording can have different meanings in different bounded contexts. The ubiquitous language is used everywhere: converstation, class names, method names and it is not replaced with synonyms.

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









