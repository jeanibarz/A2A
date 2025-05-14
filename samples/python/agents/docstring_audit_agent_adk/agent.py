import base64
import os
import json
import asyncio
from collections.abc import AsyncIterable
from typing import Any, List, Dict, Optional

# ADK Imports
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Project specific imports
from task_manager import AgentWithTaskManager # Make sure this path is correct
from common.types import FilePart # Make sure this path is correct

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai import errors
import ast

class Discrepancy(BaseModel):
    """Represents a single discrepancy found during the audit."""
    file_name: str
    discrepancy_type: str
    description: str
    suggested_change: str
    severity: str = "Medium"

class AuditReport(BaseModel):
    """Represents the full audit report."""
    discrepancies: List[Discrepancy] = Field(default_factory=list)
    summary: str = "Docstring audit complete."
    score: Optional[float] = Field(None, description="Overall docstring quality score (0-100).")
    confidence: Optional[float] = Field(None, description="Confidence level of the score (0-100), based on the number of documentable elements.")

class DocstringAuditADKAgent(AgentWithTaskManager):
    """
    ADK Agent for auditing Python code docstrings against best practices and standards.

    This agent reviews Python files and their docstrings, comparing them against
    common conventions (like PEP 257, Google style, NumPy style, etc.) and any provided
    guidelines to identify discrepancies and areas of non-adherence.
    """
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        """
        Initializes the DocstringAuditADKAgent.

        Sets up the user ID, retrieves the Google API key from environment variables,
        and initializes the Google GenAI client.

        Raises:
            ValueError: If the GOOGLE_API_KEY environment variable is not set.
        """
        self._user_id = 'remote_agent'
        self._google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self._google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self._genai_client = genai.Client(api_key=self._google_api_key)

        self._audit_model_name = 'gemini-2.5-flash-preview-04-17'

        # Defines the core prompt structure for the AI model used for docstring auditing.
        # It expects the following placeholders to be formatted:
        # - {code_analysis}: The Python code content to be audited.
        # - {guidelines}: Any specific docstring guidelines provided.
        # - {schema}: The JSON schema for the expected output format (AuditReport).
        self._system_prompt_template = f"""\
You are a helpful assistant that audits Python code for docstring quality and adherence to standards.
Your task is to review the provided Python code and its docstrings, comparing them against common Python docstring best practices (like PEP 257, Google style, NumPy style, etc.) and any specific company guidelines provided.

Identify and report discrepancies and areas of non-adherence, such as:
Only report discrepancies that are supported by a clear and good rationale based on the code and guidelines. It is not necessary to generate discrepancies if none are found.
- Missing docstrings for functions, classes, or methods.
- Inconsistencies between docstrings and the code's implementation (e.g., parameters, return values, exceptions).
- Violations of docstring formatting or style guidelines (e.g., incorrect section headers, indentation, line length).
- Lack of clarity, accuracy, or completeness in docstrings.
- Overly verbose or unnecessarily complex docstrings.

For each finding, provide:
- The file name.
- The type of discrepancy (e.g., "Missing Docstring", "Style Violation", "Inconsistency", "Insufficient Detail", "Overly Verbose").
- A description of the specific issue.
- A suggested change or action to address the discrepancy.
- A severity level ("Low", "Medium", "High").

Here is the code analysis and guidelines:
<code_analysis>
{{code_analysis}}
</code_analysis>
<guidelines>
{{guidelines}}
</guidelines>

Provide your audit findings in a JSON format matching the following schema:
{{schema}}
"""

    def get_processing_message(self) -> str:
        """
        Returns the message indicating the agent's current processing state.
        """
        return 'Auditing docstrings...'

    async def _process_file_with_google_ai(
        self,
        file_name: str,
        file_content: str,
        guidelines_content: str
    ) -> AuditReport: # Changed return type to AuditReport
        """
        Processes a single file's content using the Google GenAI model to audit docstrings.

        Formats the prompt with the file content, guidelines, and the expected schema,
        sends it to the GenAI model, and parses the JSON response into an AuditReport object.

        Args:
            file_name: The name of the file being processed.
            file_content: The content of the file as a string.
            guidelines_content: The content of the guidelines file as a string (can be empty).

        Returns:
            An AuditReport object containing the discrepancies found in the file,
            along with a calculated score and confidence. Note that potential errors
            during processing (like API errors or JSON parsing issues) are caught
            internally and reported as Discrepancy objects within the returned report's
            discrepancies list, rather than being raised as exceptions by this method.

        Notes:
            Catches and reports `google.genai.errors.APIError` as a Discrepancy.
        """
        print(f"DEBUG: Starting Google GenAI processing for file: {file_name}")

        # Parse the file content to count documentable elements
        total_documentable_elements = 0
        documented_elements = 0
        try:
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_documentable_elements += 1
                    if ast.get_docstring(node):
                        documented_elements += 1
        except SyntaxError as e:
            print(f"DEBUG: Syntax error parsing file {file_name}: {e}")
            # Create a discrepancy for syntax errors
            syntax_error_discrepancy = Discrepancy(
                file_name=file_name,
                discrepancy_type="Syntax Error",
                description=f"File has a Python syntax error: {e}",
                suggested_change="Fix the syntax error in the file.",
                severity="High"
            )
            # Return an AuditReport with the syntax error discrepancy and default scores
            return AuditReport(
                discrepancies=[syntax_error_discrepancy],
                summary=f"Audit failed due to syntax error in {file_name}.",
                score=0.0,
                confidence=0.0
            )


        prompt = self._system_prompt_template.format(
            code_analysis=file_content,
            guidelines=guidelines_content,
            schema=json.dumps(AuditReport.model_json_schema()) # Include the schema
        )
        # For debugging the prompt if issues persist:
        # print(f"DEBUG: Prompt for {file_name}:\n{prompt[:1000]}...\n")

        try:
            response = await self._genai_client.aio.models.generate_content(
                model=self._audit_model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AuditReport # Pass Pydantic model as schema
                )
            )

            print(f"DEBUG: Google GenAI response received for file: {file_name}")
            response_text = response.text
            print(f"DEBUG: ===== LLM Raw Response for {file_name} START =====")
            print(repr(response_text)) # Use repr() to see all characters, including newlines etc.
            print(f"DEBUG: ===== LLM Raw Response for {file_name} END =====")

            try:
                report_data = json.loads(response_text)
                audit_report = AuditReport(**report_data)
                for disc in audit_report.discrepancies:
                    disc.file_name = file_name
                print(f"DEBUG: Found {len(audit_report.discrepancies)} discrepancies in {file_name}")

                # Calculate score and confidence
                score = 100.0 # Start with a perfect score
                penalty = 0.0
                severity_weights = {"Low": 1, "Medium": 5, "High": 10} # Example weights

                for disc in audit_report.discrepancies:
                    penalty += severity_weights.get(disc.severity, 0)

                # Adjust score based on penalties and documented elements
                if total_documentable_elements > 0:
                    documentation_completeness = (documented_elements / total_documentable_elements) * 100
                    # Simple example: penalize missing docs and discrepancies
                    score = max(0, documentation_completeness - penalty)
                    # Confidence could be based on the number of elements, maybe capped
                    confidence = min(100.0, (total_documentable_elements / 5.0) * 20.0) # Example: 5 elements = 100% confidence

                else:
                    # Handle files with no documentable elements (e.g., config files)
                    score = 100.0 # Assume perfect score if nothing to document
                    confidence = 0.0 # Low confidence as no elements were checked

                audit_report.score = round(score, 2)
                audit_report.confidence = round(confidence, 2)

                return audit_report # Return the full AuditReport object

            except json.JSONDecodeError as jde:
                print(f"DEBUG: Failed to parse LLM JSON response for {file_name}: {jde}. Raw response: {response_text[:500]}")
                error_discrepancy = Discrepancy(
                    file_name=file_name,
                    discrepancy_type="LLM Response Parsing Error",
                    description=f"LLM response could not be parsed as JSON. Raw response extract: {response_text[:500]}",
                    suggested_change="Review LLM prompt, schema, and response. Ensure model adheres to JSON mode.",
                    severity="High"
                )
                return AuditReport( # Return AuditReport with error discrepancy
                    discrepancies=[error_discrepancy],
                    summary=f"Audit failed due to JSON parsing error for {file_name}.",
                    score=0.0,
                    confidence=0.0
                )
            except Exception as parse_e:
                print(f"DEBUG: Error creating AuditReport from parsed data for {file_name}: {parse_e}. Raw response: {response_text[:500]}")
                error_discrepancy = Discrepancy(
                    file_name=file_name,
                    discrepancy_type="AuditReport Creation Error",
                    description=f"Could not create AuditReport from parsed data. Error: {parse_e}. Raw response extract: {response_text[:500]}",
                    suggested_change="Review LLM prompt, AuditReport model, and ensure LLM output matches schema.",
                    severity="High"
                )
                return AuditReport( # Return AuditReport with error discrepancy
                    discrepancies=[error_discrepancy],
                    summary=f"Audit failed during report creation for {file_name}.",
                    score=0.0,
                    confidence=0.0
                )

        except errors.APIError as apie: # More specific error handling for Google API errors
            print(f"Google GenAI API Error for file {file_name}: Code: {apie.code}, Message: {apie.message}")
            error_description = f"Google GenAI API Error: {apie.message} (Code: {apie.code})"
            error_discrepancy = Discrepancy(
                file_name=file_name,
                discrepancy_type="Google GenAI API Error",
                description=error_description,
                suggested_change="Check API key, model name ('{self._audit_model_name}'), prompt, quotas, and Google Cloud project status.",
                severity="High"
            )
            return AuditReport( # Return AuditReport with API error discrepancy
                discrepancies=[error_discrepancy],
                summary=f"Audit failed due to Google GenAI API error for {file_name}.",
                score=0.0,
                confidence=0.0
            )
        except Exception as e: # General fallback
            print(f"Unexpected error during Google GenAI call or response handling for file {file_name}: {e}")
            error_description = f"An unexpected error occurred while processing file with Google GenAI: {e}"
            error_discrepancy = Discrepancy(
                file_name=file_name,
                discrepancy_type="General Processing Error",
                description=error_description,
                suggested_change="Review logs for details. Check API key, model name, prompt, quotas.",
                severity="High"
            )
            return AuditReport( # Return AuditReport with general error discrepancy
                discrepancies=[error_discrepancy],
                summary=f"Audit failed due to unexpected error for {file_name}.",
                score=0.0,
                confidence=0.0
            )


    async def stream(self, task_send_params: Any) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the docstring audit process and results.

        Handles incoming task parameters, extracts file data (including guidelines),
        processes Python files using the Google GenAI model, gathers the audit
        reports for each file, aggregates the results, and yields the final
        aggregated audit report as a dictionary with a 'content' key holding a JSON string.

        Args:
            task_send_params: The parameters sent with the task, expected to contain
                              file data in `task_send_params.message.parts`.

        Yields:
            An AsyncIterable of dictionaries containing the audit results.
            Each yielded dictionary has 'is_task_complete' and 'content' keys.
            'content' is a JSON string representing the AuditReport. Internal processing
            errors (e.g., file decoding, issues with AI processing) are included as
            Discrepancy entries within the yielded report's `discrepancies` list.

        Processing Details:
            - Looks for a `guidelines.md` file part for docstring guidelines.
            - Processes subsequent Python file parts using `_process_file_with_google_ai`.
            - Aggregates discrepancies, scores, and confidences from individual file reports.
            - Provides a summary when no Python files are found.
        """
        files_data_list: List[Dict[str, Any]] = []
        guidelines_content = ""

        if hasattr(task_send_params, 'message') and hasattr(task_send_params.message, 'parts'):
            for part in task_send_params.message.parts:
                if isinstance(part, FilePart):
                    print(f"DEBUG: Found FilePart: Name='{part.file.name}', HasBytes={bool(part.file.bytes)}")
                    if part.file.name and part.file.bytes:
                        files_data_list.append({
                            'name': part.file.name,
                            'content_b64': part.file.bytes
                        })
                    else:
                        print(f"DEBUG: FilePart missing name or bytes: Name='{part.file.name}', HasBytes={bool(part.file.bytes)}")
                elif hasattr(part, 'text') and part.text: # Ensure text is not None or empty
                    print(f"DEBUG: Received text part: {part.text[:100]}") # Log the text part if it's relevant
                else:
                    print(f"DEBUG: Found unexpected or empty part type in message: {type(part)}")

        if not files_data_list: # Check if any files were actually added
            no_files_summary = "No files provided for audit."
            # Check if the only part was the "doc" text and no actual files.
            if any(fd.get('name') != 'guidelines.md' for fd in files_data_list): # Should be empty if no files
                 pass # files_data_list would be empty if only "doc" text part was there and ignored.
            elif not any(hasattr(p, 'file') for p in task_send_params.message.parts if hasattr(task_send_params, 'message')):
                 no_files_summary = "No file parts found in the request."


            yield {
                'is_task_complete': True,
                'content': json.dumps({"summary": no_files_summary, "discrepancies": []}),
            }
            return

        all_discrepancies: List[Discrepancy] = []
        file_audit_reports: List[AuditReport] = [] # Collect AuditReport objects
        python_file_processing_tasks = []
        temp_python_files_to_process = []

        for file_data_item in files_data_list:
            if file_data_item.get('name') == 'guidelines.md':
                try:
                    guidelines_content = base64.b64decode(file_data_item.get('content_b64', '')).decode('utf-8')
                    print(f"DEBUG: Found and decoded guidelines file. Content length: {len(guidelines_content)}")
                except Exception as e:
                    print(f"Error decoding guidelines file: {e}")
                    all_discrepancies.append(Discrepancy( # Add decoding error as a discrepancy
                        file_name="guidelines.md",
                        discrepancy_type="File Decode Error",
                        description=f"Could not decode guidelines file: {e}",
                        suggested_change="Ensure guidelines.md is properly UTF-8 encoded and base64.",
                        severity="High"
                    ))
            elif file_data_item.get('name'): # Ensure there's a name for Python files
                temp_python_files_to_process.append(file_data_item)
            else:
                print(f"DEBUG: Skipped file data item with no name: {file_data_item}")


        print(f"DEBUG: Processing {len(temp_python_files_to_process)} Python files.")

        for py_file_data in temp_python_files_to_process:
            file_name = py_file_data.get('name') # Already know it has a name
            file_content_b64 = py_file_data.get('content_b64', '')
            try:
                file_content_str = base64.b64decode(file_content_b64).decode('utf-8')
                python_file_processing_tasks.append(
                    self._process_file_with_google_ai(
                        file_name=file_name,
                        file_content=file_content_str,
                        guidelines_content=guidelines_content
                    )
                )
            except Exception as e:
                print(f"Error decoding Python file {file_name}: {e}")
                all_discrepancies.append(Discrepancy( # Add decoding error as a discrepancy
                    file_name=file_name,
                    discrepancy_type="File Decode Error",
                    description=f"Could not decode Python file content: {e}",
                    suggested_change="Ensure file is properly UTF-8 encoded and base64.",
                    severity="High"
                ))

        if python_file_processing_tasks:
            results_from_tasks = await asyncio.gather(*python_file_processing_tasks)
            for report_for_file in results_from_tasks: # Process AuditReport objects
                if isinstance(report_for_file, AuditReport):
                    file_audit_reports.append(report_for_file)
                    all_discrepancies.extend(report_for_file.discrepancies)
                elif isinstance(report_for_file, list) and all(isinstance(d, Discrepancy) for d in report_for_file):
                     # Handle the case where _process_file_with_google_ai might still return a list of discrepancies in some error paths
                     all_discrepancies.extend(report_for_file)
                     # Create a minimal report for scoring purposes if needed, or just log a warning
                     print(f"DEBUG: _process_file_with_google_ai returned a list of discrepancies instead of AuditReport for a file.")
                else:
                    print(f"DEBUG: Unexpected return type from _process_file_with_google_ai: {type(report_for_file)}")
                    # Handle unexpected return types if necessary

        # Aggregate scores and confidences
        total_score = 0.0
        total_confidence = 0.0
        valid_reports_count = 0

        for report in file_audit_reports:
            if report.score is not None and report.confidence is not None:
                total_score += report.score
                total_confidence += report.confidence
                valid_reports_count += 1

        overall_score = total_score / valid_reports_count if valid_reports_count > 0 else None
        overall_confidence = total_confidence / valid_reports_count if valid_reports_count > 0 else None

        final_summary = f"Docstring audit completed. Found {len(all_discrepancies)} total discrepancies across {len(temp_python_files_to_process)} Python files."
        if not temp_python_files_to_process:
             final_summary = "No Python files provided for audit."
             if guidelines_content:
                 final_summary += " Guidelines were processed."
             if any(d.file_name == "guidelines.md" and d.severity=="High" for d in all_discrepancies):
                  final_summary += " (Error processing guidelines file)."


        final_report = AuditReport(
            discrepancies=all_discrepancies,
            summary=final_summary,
            score=overall_score,
            confidence=overall_confidence
        )

        yield {
            'is_task_complete': True,
            'content': json.dumps(final_report.model_dump()), # Serialize to a valid JSON string
        }
