\# System Architecture Overview: Global Financial Core (GFC)



\## 1. Executive Summary

The Global Financial Core (GFC) is a distributed, multi-tenant architecture designed to handle high-throughput financial transactions. The system prioritizes \*\*eventual consistency\*\* for ledgering and \*\*strong consistency\*\* for invoice generation. It is built on a microservices mesh to allow independent scaling of billing and payment processing during peak loads.



\## 2. System Topology

The architecture follows a \*\*Hexagonal Architecture\*\* (Ports and Adapters) pattern within each microservice to decouple business logic from external dependencies like databases and message brokers.



\### Core Service Definitions

| Service Name | Primary Responsibility | Data Store |

| :--- | :--- | :--- |

| `billing-service` | Invoice orchestration, tax calculation, and PDF generation. | PostgreSQL (Relational) |

| `payments-service` | Integration with PSPs (Stripe, Adyen), vaulting, and 3DS auth. | DynamoDB (Key-Value) |

| `ledger-service` | Double-entry bookkeeping and immutable audit trails. | Aurora Serverless |

| `reporting-service` | Materialized views for financial analytics and EOM reports. | ClickHouse (OLAP) |


---


\## 3. Transactional Workflows



\### 3.1 End-to-End Billing Lifecycle

The transition from an Order to a settled Invoice follows a choreographed saga pattern:

1\. \*\*Trigger:\*\* The `order-service` emits an `OrderStateChanged` event with the status `COMPLETED`.

2\. \*\*Validation:\*\* `billing-service` consumes the event and validates line-item integrity against the Global Product Catalog.

3\. \*\*Taxation:\*\* The service calls the `tax-engine` sidecar to calculate VAT/Sales Tax based on the customer’s `jurisdiction\_id`.

4\. \*\*Persistence:\*\* The invoice is saved with a `DRAFT` status, and a unique `invoice\_correlation\_id` is generated.

5\. \*\*Broadcast:\*\* An `InvoiceCreated` event is published to the `fincore.events` Kafka topic.



\### 3.2 Payment Processing \& Settlement

1\. \*\*Ingestion:\*\* `payments-service` identifies the `payment\_method\_id` from the customer profile.

2\. \*\*Authorization:\*\* A synchronous mTLS call is made to the external Payment Service Provider (PSP).

3\. \*\*State Machine:\*\* - If `AUTHORIZED`, the service emits `PaymentConfirmed`.

&nbsp;   - If `DECLINED`, the service triggers a `PaymentFailed` event, initiating the \*\*Dunning Cycle\*\*.

4\. \*\*Ledgering:\*\* `ledger-service` performs an atomic transaction:

&nbsp;   - \*\*DEBIT\*\* Customer Accounts Receivable.

&nbsp;   - \*\*CREDIT\*\* Revenue Account.


---


\## 4. Communication Patterns



\### 4.1 Asynchronous Messaging (Event-Driven)

We utilize \*\*Apache Kafka\*\* as the backbone for inter-service communication.

\* \*\*Topic Strategy:\*\* One topic per domain entity (e.g., `invoices`, `payments`).

\* \*\*Guarantees:\*\* At-least-once delivery with idempotent consumers implemented via `idempotency\_key` tracking in Redis.



\### 4.2 Synchronous APIs (Internal)

\* \*\*Protocol:\*\* gRPC is used for high-performance service-to-service communication (e.g., `billing` calling `ledger` for real-time balance checks).

\* \*\*Documentation:\*\* All RESTful endpoints are exposed via OpenAPI 3.0 specs hosted on the internal Developer Portal.



---



\## 5. Resilience \& Fault Tolerance



\### 5.1 Retry Strategy

The system implements a \*\*Decorrelated Jitter\*\* backoff strategy for all external API integrations. This prevents "thundering herd" issues when a downstream PSP recovers from an outage.



\### 5.2 Dead Letter Queues (DLQ)

Messages that fail processing after 5 attempts are routed to a `service-name.dlq`.

\* \*\*Alerting:\*\* PagerDuty is triggered if the DLQ depth exceeds 50 messages.

\* \*\*Recovery:\*\* Admin CLI tools allow for "replay" functionality once the root cause is resolved.



---



\## 6. Security \& Compliance

\* \*\*Data at Rest:\*\* All databases utilize AES-256 encryption.

\* \*\*Data in Transit:\*\* TLS 1.3 is enforced for all traffic.

\* \*\*PCI-DSS:\*\* The `payments-service` is isolated in a separate VPC (Virtual Private Cloud) with restricted egress to minimize the compliance scope.

\* \*\*Authentication:\*\* Service-to-service auth is handled via \*\*SPIFFE/SPIRE\*\* identities.

