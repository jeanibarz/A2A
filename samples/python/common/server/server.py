import json
import logging

from collections.abc import AsyncIterable
from typing import Any

from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response # Ensure Response is imported


from common.server.task_manager import TaskManager
from common.types import (
    A2ARequest,
    AgentCard,
    CancelTaskRequest,
    GetTaskPushNotificationRequest,
    GetTaskRequest,
    InternalError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCResponse,
    SendTaskRequest,
    SendTaskStreamingRequest,
    SetTaskPushNotificationRequest,
    TaskResubscriptionRequest,
)


logger = logging.getLogger(__name__)


class A2AServer:
    def __init__(
        self,
        host='0.0.0.0',
        port=5000,
        endpoint='/',
        agent_card: AgentCard = None,
        task_manager: TaskManager = None,
    ):
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.task_manager = task_manager
        self.agent_card = agent_card

        # Initialize Starlette app with default middleware
        self.app = Starlette()
        self.app.add_route(
            self.endpoint, self._process_request, methods=['POST']
        )
        self.app.add_route(
            '/.well-known/agent.json', self._get_agent_card, methods=['GET']
        )

    def start(self):
        if self.agent_card is None:
            raise ValueError('agent_card is not defined')

        if self.task_manager is None:
            raise ValueError('request_handler is not defined')

        import uvicorn

        # Run Uvicorn with the Starlette app.
        # Request body size limit is handled manually in _process_request.
        uvicorn.run(self.app, host=self.host, port=self.port)

    def _get_agent_card(self, request: Request) -> JSONResponse:
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    async def _process_request(self, request: Request):
        try:
            # Define maximum allowed request body size (e.g., 10MB)
            MAX_BODY_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

            # Read the raw request body
            raw_body = await request.body()

            # Check if the body size exceeds the limit
            if len(raw_body) > MAX_BODY_SIZE_BYTES:
                # Raise an HTTPException that will be caught by _handle_exception
                raise StarletteHTTPException(
                    status_code=400,
                    detail=f"Request body size ({len(raw_body)} bytes) exceeds limit ({MAX_BODY_SIZE_BYTES} bytes)."
                )

            # Parse the raw body as JSON
            body_content = json.loads(raw_body)

            # Validate the JSON content against the expected schema
            json_rpc_request = A2ARequest.validate_python(body_content)

            if isinstance(json_rpc_request, GetTaskRequest):
                result = await self.task_manager.on_get_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskRequest):
                result = await self.task_manager.on_send_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskStreamingRequest):
                result = await self.task_manager.on_send_task_subscribe(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, CancelTaskRequest):
                result = await self.task_manager.on_cancel_task(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, SetTaskPushNotificationRequest):
                result = await self.task_manager.on_set_task_push_notification(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, GetTaskPushNotificationRequest):
                result = await self.task_manager.on_get_task_push_notification(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, TaskResubscriptionRequest):
                result = await self.task_manager.on_resubscribe_to_task(
                    json_rpc_request
                )
            else:
                logger.warning(
                    f'Unexpected request type: {type(json_rpc_request)}'
                )
                raise ValueError(f'Unexpected request type: {type(request)}')

            return self._create_response(result)

        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> JSONResponse:
        request_id = None # Placeholder, ideally extracted from request if available
        
        if isinstance(e, StarletteHTTPException):
            if e.status_code == 400 and "Maximum request body size limit exceeded" in e.detail:
                logger.warning(f'Request body size limit exceeded: {e.detail}')
                json_rpc_error = InvalidRequestError(
                    message="Request entity too large.",
                    data={"details": e.detail, "limit": "Starlette's default is 1MB for JSON payloads unless configured otherwise."}
                )
            elif e.status_code == 400 and "Request body larger than max_body_size limit" in e.detail: # Uvicorn related
                logger.warning(f'Request body size limit exceeded (Uvicorn): {e.detail}')
                json_rpc_error = InvalidRequestError(
                    message="Request entity too large.",
                    data={"details": e.detail, "limit": "Check Uvicorn's max_body_size configuration."}
                )
            else:
                logger.warning(f'Starlette HTTP Exception: {e.detail} (status_code: {e.status_code})')
                json_rpc_error = InvalidRequestError(message=e.detail or "Invalid request due to HTTP error.")
        elif isinstance(e, json.decoder.JSONDecodeError):
            logger.warning(f'JSON decode error: {e}')
            json_rpc_error = JSONParseError(data={"details": str(e)})
        elif isinstance(e, ValidationError):
            logger.warning(f'Pydantic validation error: {e.errors()}')
            try:
                error_data = json.loads(e.json())
            except json.JSONDecodeError:
                error_data = {"raw_errors": str(e.errors())} # Fallback if e.json() is not valid JSON
            json_rpc_error = InvalidRequestError(data=error_data)
        else:
            logger.error(f'Unhandled exception: {e}', exc_info=True)
            json_rpc_error = InternalError(data={"details": f"An unexpected error occurred: {type(e).__name__}"})

        response = JSONRPCResponse(id=request_id, error=json_rpc_error)
        
        # Determine status code for the HTTP response
        if isinstance(json_rpc_error, InternalError):
            status_code = 500  # Internal server error
        elif hasattr(e, 'status_code') and isinstance(e.status_code, int) and 400 <= e.status_code < 600:
            # If the original exception (e.g., StarletteHTTPException) had a valid HTTP status code, use it.
            status_code = e.status_code
        elif isinstance(e, (json.decoder.JSONDecodeError, ValidationError)):
            status_code = 400 # Bad request for parsing or validation errors
        else:
            # Default for other client-side errors if not an InternalError and no specific status from e
            status_code = 400 # Default bad request

        return JSONResponse(
            response.model_dump(exclude_none=True), status_code=status_code
        )

    def _create_response(
        self, result: Any
    ) -> JSONResponse | EventSourceResponse:
        if isinstance(result, AsyncIterable):

            async def event_generator(result_iterable: AsyncIterable) -> AsyncIterable[dict[str, str]]:
                async for item in result_iterable:
                    yield {'data': item.model_dump_json(exclude_none=True)}

            return EventSourceResponse(event_generator(result)) # Corrected: pass result_iterable
        if isinstance(result, JSONRPCResponse):
            return JSONResponse(result.model_dump(exclude_none=True))
        
        # Fallback for unexpected result types: Log and return a structured internal error.
        logger.error(f'Unexpected result type in _create_response: {type(result)}')
        error_payload = InternalError(data={"details": f"Server error: Handler returned unexpected result type {type(result).__name__}"})
        response_payload = JSONRPCResponse(id=None, error=error_payload) # Assuming no request ID context here
        return JSONResponse(response_payload.model_dump(exclude_none=True), status_code=500)
