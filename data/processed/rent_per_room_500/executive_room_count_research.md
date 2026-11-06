# Executive flat room-count audit

The 8,234 `executive` rental records cover 1,867 unique town/street/block/model/area groups. The source rental table has no unit or stack number. `executive_address_audit.csv` lists these groups and leaves `VERIFIED_ROOM_COUNT` blank.

HDB describes the standard executive flat as having a master bedroom, two additional bedrooms, and extra study or family space. Its renovation guidance separately refers to an executive apartment **with four bedrooms**, confirming that a single bedroom count should not be assigned to every executive flat. HDB's public Property Information dataset gives an `Exec Sold` count by block, but no unit bedroom count or floor plan. HDB's unit floor-plan service requires the particular unit address and sells plans individually. Therefore the rental table's block address does not identify an exact physical room count for each executive approval.

The existing `cleaned.csv` and `removed_below_500.csv` use the *numeric HDB flat-type label* as the divisor for 1- to 5-room types. That label is not a literal bedroom count: HDB lists both 4-room and 5-room flats as typically having three bedrooms. Executive rows remain in `executive_unclassified.csv`; no executive rent-per-room value or threshold decision was fabricated.

Official sources:

- [HDB flat-type features](https://www.hdb.gov.sg/buying-a-flat/bto-sbf-and-open-booking-of-flats/finding-a-new-flat/types-of-flats)
- [HDB renovation guidance distinguishing four-bedroom executive apartments](https://www.hdb.gov.sg/managing-my-home/renovation-and-maintenance/renovation/renovation-guidelines/air-conditioner-installation-works)
- [HDB Property Information block-level schema](https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view)
- [HDB unit floor-plan service](https://www.hdb.gov.sg/buy-floor-plan-of-hdb-flat/get-help)
