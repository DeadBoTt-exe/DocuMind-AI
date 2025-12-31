\# Billing Service Documentation



\## Service Responsibility

The billing-service is responsible for invoice generation, tax calculation, discounts, and final invoice persistence.




\## Invoice Creation

Invoices are created when an order reaches the COMPLETED state. Each invoice contains:

\- Line items

\- Subtotal

\- Tax breakdown

\- Total payable amount



\## GST Calculation

GST is calculated during invoice finalization. The billing-service applies GST based on:

\- Customer location

\- Product tax category

\- Applicable GST slabs



The GST logic is implemented as follows:

\- CGST and SGST are applied for intra-state transactions.

\- IGST is applied for inter-state transactions.



Tax rates are fetched from the tax-config table and cached in memory.



\## Discounts

Discounts are applied before tax calculation. Supported discount types:

\- Flat discounts

\- Percentage discounts

\- Promotional coupon discounts



\## Invoice States

\- DRAFT

\- FINALIZED

\- PENDING\_PAYMENT

\- PAID

\- CANCELLED



