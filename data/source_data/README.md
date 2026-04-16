# implicit_reference_dataset

This repository contains the generated implicit reference datasets.

## Datasets

1. **Implicit references from short text segments**:
   - Derived from veterans interview transcripts converted to recollections.

2. **Twitter dataset**:
   - Taken from [Implicit Entity Recognition and Linking in Tweets](https://github.com/HawreH/Implicit-Entity-Recognition-and-Linking-in-Tweets-Resources-and-Dataset).

## Dataset Columns

- **uid**: Unique sample ID.
- **source**: Unique source ID (Tweet ID or veteran name).
- **SQN**: Non-empty only for reminiscence data. Text Sequence Number (sequence number in series of text for the same source).
- **text**: Tweet text or rephrased reminiscence statement.
- **origin**: Non-empty only for reminiscence data. Original text before any rephrasing.
- **entity**: Implicit entity, all lower case.
- **entity_type**: Implicit entity class type: people, places, locations, organizations, events, or dates.
- **links**: Tweet URL or veteran interview transcript URL.
