Below is a **clean, system-level design** for a **DeepAgent-based Intelligent File Understanding and Reconstruction System**, explicitly targeting **complex heterogeneous file inputs** (PDFs, images, scans, mixed folders) and **user-goal–driven restructuring** (OCR, renaming, hierarchy reconstruction, and task-oriented organization such as reimbursement submission).

This design is intentionally aligned with the **orchestrated deepagent prompt style** you provided: deterministic workflow, strict role separation, artifact-oriented outputs, and tool-mediated cognition.

---

# Intelligent File Understanding & Reconstruction DeepAgent

_(Goal-driven, auditable, workflow-oriented design)_

---

## 1. System Objective (Problem Framing)

**Goal**  
Transform a large, unstructured collection of files into a **task-aligned, semantically organized file system**, driven by:

- User intent (explicit request + implicit constraints)
    
- Document semantics (OCR + layout + metadata)
    
- External system requirements (e.g., reimbursement platform formats)
    

**Non-goals**

- Free-form chat-based assistance
    
- One-shot “smart renaming” without traceability
    
- Black-box reorganization without explanation or review
    

---

## 2. High-Level Architecture

```
User Request + Input Files
        │
        ▼
┌─────────────────────────────┐
│  Orchestrator DeepAgent     │
│  (Stateful Controller)     │
└─────────────┬──────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
File Understanding   User-Intent Modeling
Sub-Agents           Sub-Agents
(OCR / parsing)      (task constraints)
    │                   │
    └─────────┬─────────┘
              ▼
   Semantic Normalization Layer
   (document schema inference)
              │
              ▼
   Hierarchy Reconstruction Engine
   (rename, foldering, grouping)
              │
              ▼
   Output Artifacts + User Review Surface
```

---

## 3. Core Design Principles

### 3.1 Workflow Determinism

Every execution follows a **fixed lifecycle**:

1. Intent capture
    
2. File inventory
    
3. Semantic extraction
    
4. Task schema alignment
    
5. Structural reconstruction
    
6. Review & export
    

No phase is skipped.

---

### 3.2 Separation of Cognition

|Role|Responsibility|
|---|---|
|Orchestrator|Planning, delegation, synthesis, final decisions|
|File Sub-Agents|OCR, layout parsing, metadata extraction|
|Intent Sub-Agents|Understand user goal and system constraints|
|Structuring Sub-Agent|Propose hierarchy & naming schemes|
|Validator|Check compliance with user/system rules|

Sub-agents **never reorganize files directly**.

---

### 3.3 Artifact-First Design

All key intermediate states are persisted:

|Artifact|Purpose|
|---|---|
|`/user_intent.md`|Canonical task definition|
|`/file_inventory.json`|Raw file index|
|`/doc_semantics.json`|OCR + parsed semantics|
|`/proposed_structure.md`|Human-readable plan|
|`/final_mapping.json`|Old → new mapping|
|`/audit_log.md`|Decision trace|

This enables **auditing, rollback, and human-in-the-loop review**.

---

## 4. Detailed Workflow Design

---

## Phase 1 — User Intent Modeling

### Input

- Natural language request  
    _Example:_  
    “These are receipts and invoices from business trips. I need to submit them into the company reimbursement system, grouped by trip and expense type.”
    

### Orchestrator Actions

- Save request to `/user_intent.md`
    
- Delegate **Intent Sub-Agent**
    

### Intent Sub-Agent Output

Structured intent model:

```json
{
  "task_type": "reimbursement",
  "grouping_keys": ["trip", "date", "expense_type"],
  "required_fields": [
    "amount",
    "currency",
    "invoice_date",
    "vendor",
    "tax_id"
  ],
  "output_constraints": {
    "one_receipt_per_entry": true,
    "pdf_preferred": true,
    "naming_pattern": "{date}_{vendor}_{amount}"
  }
}
```

---

## Phase 2 — File Inventory & Classification

### Orchestrator

- Enumerate all files
    
- Persist `/file_inventory.json`
    
- Delegate **File Understanding Sub-Agent**
    

### File Sub-Agent Tasks

- Detect file type (PDF / image / scan)
    
- Run OCR if necessary
    
- Extract layout-aware text
    
- Detect duplicates / low-quality scans
    

**No interpretation yet. Only extraction.**

---

## Phase 3 — Semantic Normalization

### Purpose

Convert raw OCR/text into **canonical document schemas**.

### Normalized Schema Example

```json
{
  "doc_id": "IMG_0231.jpg",
  "doc_type": "receipt",
  "vendor": "Starbucks",
  "date": "2024-11-03",
  "amount": 42.50,
  "currency": "USD",
  "tax": 3.20,
  "confidence": {
    "date": 0.98,
    "amount": 0.95
  }
}
```

Stored in `/doc_semantics.json`.

---

## Phase 4 — Hierarchy Reconstruction Planning

### Structuring Sub-Agent

Inputs:

- `/user_intent.md`
    
- `/doc_semantics.json`
    

Output:

- A **proposed hierarchy plan**, not execution.
    

Example:

```text
Reimbursement/
├── 2024-11_BusinessTrip_Shanghai/
│   ├── Meals/
│   │   ├── 2024-11-03_Starbucks_42.50.pdf
│   ├── Transport/
│   └── Hotel/
└── Summary.xlsx
```

Saved as `/proposed_structure.md`.

---

## Phase 5 — Validation & Constraint Checking

Validator Sub-Agent checks:

- Missing required fields
    
- Low OCR confidence
    
- Violations of naming rules
    
- Multiple receipts merged incorrectly
    

Produces `/validation_report.md`.

---

## Phase 6 — Execution (Optional / Confirmed)

Only after user approval:

- Rename files
    
- Reorganize folders
    
- Generate auxiliary outputs:
    
    - CSV / Excel for upload
        
    - ZIP per reimbursement batch
        

Final mapping persisted:

```json
{
  "IMG_0231.jpg": "Reimbursement/2024-11_BusinessTrip_Shanghai/Meals/2024-11-03_Starbucks_42.50.pdf"
}
```

---

## 5. User Interaction Model

### Review Surfaces

- Diff-style preview (old → new)
    
- Confidence flags (highlight uncertain OCR)
    
- Editable naming templates
    
- “Approve / Reject / Adjust” loop
    

### User Control

- Can override grouping logic
    
- Can re-run only specific phases
    
- Can export without modifying originals
    

---

## 6. Why This Design Works Well for Your Context

This architecture:

- Scales to **thousands of documents**
    
- Is **fully auditable**, critical for finance/legal workflows
    
- Aligns with your **Finder Agent / New-Era File System** vision
    
- Naturally integrates with:
    
    - RAG indexing
        
    - Enterprise systems
        
    - Multi-agent LangGraph / DeepAgents backends
        

It treats file organization not as a UX trick, but as a **goal-conditioned cognitive task**.

---

## 7. Natural Extensions (If You Want Next)

Possible next steps include:

- Formalizing this as a **LangGraph StateMachine**
    
- Designing the **prompt contracts** for each sub-agent
    
- Adding **policy hooks** (e.g., finance compliance rules)
    
- Mapping to your **SQLite-VEC / FAISS knowledge layer**
    
- Introducing **long-term memory for vendors / trips**
    

If you want, I can proceed with **prompt templates**, **state diagrams**, or a **concrete LangGraph implementation sketch**.