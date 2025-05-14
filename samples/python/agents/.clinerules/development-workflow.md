## Brief overview
This rule outlines preferences for the development workflow and communication style when working with the user, based on observed interactions.

## Development workflow
  - The user prefers to execute the agent or application themselves after code changes. Do not attempt to run the agent using `execute_command` unless explicitly instructed.
  - After making code modifications, wait for the user to run the application and provide feedback on the result.
  - When modifying files, use `replace_in_file` for targeted changes. If `replace_in_file` fails multiple times due to content mismatches, use `write_to_file` as a fallback to write the complete file content.

## Debugging approach
  - When encountering errors, prioritize detailed technical analysis to understand the root cause.
  - Be prepared to add logging or print statements to inspect intermediate outputs and data, such as raw LLM responses, to aid debugging.
  - Consider alternative approaches or workarounds when blocked by external factors (e.g., framework limitations).

## Communication style
  - Maintain clear, technical, and detailed communication, similar to the style of the user's feedback.
  - Provide specific code snippets and explanations when discussing issues or proposed changes.
