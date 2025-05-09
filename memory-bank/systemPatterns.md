# System Patterns: OLMoCR

## Architectural Overview
1. **Modular Pipeline Design**:
   - Components for each processing stage (ingestion, analysis, filtering)
   - Clear interfaces between modules
   - Configurable processing flows

2. **Document Processing Flow**:
   PDF → Layout Analysis → Content Extraction → Quality Assessment → Filtering → Output

## Key Components
1. **pipeline.py**: Main processing pipeline
2. **image_utils.py**: Image processing utilities
3. **metrics.py**: Quality metrics calculation
4. **filter/**: Document filtering modules
5. **bench/**: Benchmarking framework

## Design Patterns
1. **Pipeline Pattern**:
   - Sequential processing stages
   - Each stage operates on standardized document representation
   - Easy to add/remove processing steps

2. **Strategy Pattern**:
   - Different implementations for layout analysis
   - Configurable quality metrics
   - Pluggable filtering criteria

3. **Observer Pattern**:
   - Progress monitoring
   - Quality metric collection
   - Event logging

4. **Wrapper Pattern**:
   - Sync process_pdf_file() wraps async process_pdf()
   - Enables both CLI and HTTP interfaces
   - Maintains single implementation of core logic

## Data Structures
1. **Document Representation**:
   - Pages
   - Blocks (text, images, tables)
   - Layout metadata
   - Quality scores

2. **Processing Context**:
   - Current document state
   - Configuration
   - Metrics collection
