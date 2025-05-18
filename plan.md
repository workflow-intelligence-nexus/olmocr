# Plan to Align API PDF Processing with `processPDF.sh`

**Goal:** Ensure the FastAPI `/process-pdf` endpoint (via `http_service.py` and `startLocalServer.sh`) correctly processes files like `scans/testme1.pdf` and achieves the same successful text extraction results as when using the `./processPDF.sh` command directly.

**Background:**
The API currently fails for `scans/testme1.pdf`, resulting in a 500 error. Debug logs indicate that the SGLang model returns `None` for `natural_text` for all pages of this PDF when called via the API. In contrast, `./processPDF.sh` successfully extracts text from the same file.

**Key Discrepancies Identified (from logs):**
1.  **SGLang Server `dtype`**:
    *   API's SGLang (started by `startLocalServer.sh`): `half` (float16)
    *   `processPDF.sh`'s SGLang (when it starts its own instance): `bfloat16`
2.  **SGLang Server `disable_cuda_graph`**:
    *   API's SGLang: `True` (CUDA graph disabled)
    *   `processPDF.sh`'s SGLang: `False` (CUDA graph enabled)
3.  **Anchor Text**: While logs showed a short anchor text (52 chars) for `scans/testme1.pdf` in both contexts, the different SGLang configurations might handle this differently. The successful `processPDF.sh` run *did* produce text despite this short anchor.

**Hypothesis:** The primary reason for the discrepancy is the difference in SGLang server configurations (`dtype` and `disable_cuda_graph`) between the two execution paths. These differences likely cause the SGLang model to behave differently, especially in its ability to extract text when anchor text might be suboptimal.

---

## Action Plan:

**Instructions for the AI Coder:**
*   Follow each step sequentially.
*   After each "Action" step that involves a code change, wait for the user to confirm the change was applied and to provide new test results/logs.
*   Update the "Status" and "Observations/Results" for each step based on user feedback.
*   If a step resolves the issue, mark subsequent contingency steps as "Skipped".

---

### **Step 1: Align SGLang Server Parameters in `startLocalServer.sh`**

*   **Description:** Modify `startLocalServer.sh` to launch its SGLang server with `dtype=auto` (which should resolve to `bfloat16` as seen in `processPDF.sh` logs) and to *enable* CUDA graphs (by removing `--disable-cuda-graph`).
*   **Files to Modify:** `startLocalServer.sh`
*   **Specific Changes:**
    1.  In the `python -m sglang.launch_server ...` command:
        *   Change `--dtype half` to `--dtype auto`.
        *   Remove the `--disable-cuda-graph` line.
*   **Status:** `PENDING`
*   **Action by AI Coder:**
    ```xml
    <replace_in_file>
    <path>startLocalServer.sh</path>
    <diff>
    <<<<<<< SEARCH
      --port $SGLANG_PORT \
      --device cuda \
      --dtype half \
      --mem-fraction-static 0.80 \
      --disable-cuda-graph \
      --host 0.0.0.0 &
    SGLANG_PID=$!
    =======
      --port $SGLANG_PORT \
      --device cuda \
      --dtype auto \
      --mem-fraction-static 0.80 \
      --host 0.0.0.0 &
    SGLANG_PID=$!
    >>>>>>> REPLACE
    </diff>
    </replace_in_file>
    ```
*   **Expected Outcome:** The SGLang server started by `startLocalServer.sh` will now use `bfloat16` and have CUDA graphs enabled, matching the configuration of the SGLang server implicitly started by `processPDF.sh`.
*   **Observations/Results:** (To be filled in by AI Coder after user feedback)

---

### **Step 2: Test API with Aligned SGLang Configuration**

*   **Description:** After the user restarts `startLocalServer.sh` (which will use the updated SGLang parameters from Step 1), they will re-test processing `scans/testme1.pdf` via the `/process-pdf` API endpoint.
*   **Files to Modify:** None.
*   **Status:** `PENDING`
*   **Action by AI Coder:** (Wait for user to perform test and provide logs)
    ```xml
    <ask_followup_question>
    <question>I've prepared the change for `startLocalServer.sh` to align SGLang parameters. After I apply it and you restart the server, please test `scans/testme1.pdf` via the API again and provide the new console logs from `startLocalServer.sh`, focusing on the API_DEBUG messages related to anchor text and per-page natural_text.</question>
    </ask_followup_question>
    ```
*   **Expected Outcome:** The API successfully processes `scans/testme1.pdf` and extracts text, similar to `processPDF.sh`. The `API_DEBUG` logs should show non-empty `natural_text` for the pages and a non-zero `final document_text length`.
*   **Observations/Results:** (To be filled in by AI Coder after user feedback)
    *   Did the API call succeed?
    *   What was the `anchor_text length`?
    *   What was the `natural_text` for each page?
    *   What was the `final document_text length`?
    *   What was the HTTP response from the API?

---

### **Step 3: (Contingency) Investigate Anchor Text Generation Differences**

*   **Description:** If Step 2 does not resolve the issue (i.e., `natural_text` is still `None` or empty for pages of `scans/testme1.pdf` via API), this step will focus on why `get_anchor_text` might be producing different or ineffective results in the API context compared to the `processPDF.sh` context, even if the SGLang server is identically configured.
*   **Files to Modify:** Potentially `olmocr/pipeline.py` (to add more detailed logging for `anchor_text` content).
*   **Status:** `PENDING`
*   **Action by AI Coder:** (To be determined based on results of Step 2)
    *   If needed, add logging for the first ~200 characters of `anchor_text` in `build_page_query`.
    *   Guide the user on how to capture this detailed anchor text from both API and `processPDF.sh` runs for comparison.
*   **Expected Outcome:** Identify if the content of `anchor_text` (not just its length) differs significantly between the two execution paths for `scans/testme1.pdf`.
*   **Observations/Results:** (To be filled in by AI Coder after user feedback)

---

### **Step 4: (Contingency) Examine Subprocess Environment for `get_anchor_text`**

*   **Description:** If anchor text content *is* different or if SGLang (even when identically configured) still fails with the API-generated anchor text, investigate potential environmental differences affecting the tools used by `get_anchor_text` (e.g., `pdfreport`, `pdftotext`) when run from a `ProcessPoolExecutor` managed by a Uvicorn worker vs. a direct CLI script.
*   **Files to Modify:** None initially; focus on analysis.
*   **Status:** `PENDING`
*   **Action by AI Coder:** (To be determined based on results of Step 3)
    *   Discuss potential differences in `PATH`, library paths (`LD_LIBRARY_PATH`), or other environment variables available to subprocesses.
    *   Consider if `get_anchor_text` needs to be called with more explicit paths or environment settings when run via the API.
*   **Expected Outcome:** Identify and remediate any environmental factors causing `get_anchor_text` to behave differently.
*   **Observations/Results:** (To be filled in by AI Coder after user feedback)

---
