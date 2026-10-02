# Reusing Edesia for the kitchen demo

`DasilvaKareem/CateringAssistant` was inspected locally as the existing application.
Its source has not been copied into this repository.

| Existing module | Reuse |
| --- | --- |
| `src/lib/menu-types.ts` | Recipe ingredient mappings and `scaledIngredients` |
| `src/lib/inventory-data.ts` | Inventory lots, stock consumption, cost records |
| `src/lib/shift-vision.ts` | Video service caller and review integration |
| `src/lib/shifts-types.ts` | Detection and timeline types |
| `src/lib/replay-doc.ts` | Playable video with a timestamped deduction timeline |
| `src/lib/square-data.ts`, `toast-pos-data.ts`, `clover-data.ts` | Existing POS data integration |

The application calls an external Cloud Run vision service configured by
`VISION_SERVICE_URL`. That service implementation is absent from the inspected
repository. Cosmos inference needs a new service implementation or access to the
existing service source; changing the Gemini model name alone cannot replace it.

## Evidence and consumption

Cosmos output describes visible actions. Keep these observations separate from a
recipe/POS consumption proposal. An observation needs the source clip identity,
timestamp range, ingredient identity, action, evidence, and uncertainty.

Expected ingredient usage comes from recipe quantities multiplied by fulfilled
portions, with recipe yield/waste allowances clearly identified. Existing
`scaledIngredients` already includes `wasteFactor`; that allowance must not be
presented as measured waste.

Recipe/POS deductions are expected usage. Footage can support or question those
records but does not establish weight or volume without an independently
validated measurement. Discarding an ingredient can justify a possible waste
flag; apparent portion differences and preparations without matched orders
remain review candidates.

## Changes needed before stock writes

- Existing shift ingestion invokes validation and can deduct stock immediately.
  Keep Cosmos observations pending until they are reconciled and reviewed.
- Existing detection IDs are scoped to a generated shift ID. Uploading the same
  file as a new shift can therefore create new consumption records. Use the
  clip SHA256 plus an event identity, scoped to the business and source camera.
- Existing order completion and video detections both have stock deduction paths.
  Match them to one consumption record; adding video evidence to an already
  consumed POS/order record must not debit ingredients again.
- Guard stock changes and the idempotency record together in a database
  transaction. Writing a deterministic consumption event ID after changing stock
  does not by itself prevent races or partial failures.

For the demo, use one known recipe and a small POS fixture. Show a preparation,
a pickup followed by return without visible use, and a repeated upload. Evidence
links should seek the video to the corresponding timestamp. Repeated uploads
should reuse the existing result and leave stock unchanged.
