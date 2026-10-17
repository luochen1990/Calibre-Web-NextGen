### Fixed

- Reflow keeps legible scanned text selectable when a model reading is unavailable, including conversions without a key or after a spending limit. Local OCR remains visibly unverified: each paragraph links to its original detail, and the complete original page is available beside it. An existing OCR text layer cannot certify those words.
- Local recognition uses bounded rasters, execution time and output, records word coordinates and engine scores, and keeps its temporary receipts in the conversion's owned scratch. Weak or garbled paragraphs retain source images independently of healthy text.
- An unproved footnote association no longer removes otherwise supported body prose. Only the uncertain marker or note receives source disclosure.
- The joint text-quality check scales with passage length, including passages shorter than forty words and image descriptions.
