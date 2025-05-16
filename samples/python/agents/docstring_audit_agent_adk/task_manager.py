import json
import logging

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Any

from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    InternalError,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.genai import types


logger = logging.getLogger(__name__)


# TODO: Move this class (or these classes) to a common directory
class AgentWithTaskManager(ABC):
    """
    Abstract base class defining the interface for agents integrated with the task manager.

    Agents that are intended to be managed by the AgentTaskManager should inherit
    from this class and implement the required abstract methods.
    """
    @abstractmethod
    def get_processing_message(self) -> str:
        """
        Returns a string indicating the agent's current processing status or message for the task.

        This message can be used to provide updates to the user during long-running tasks.
        """
        pass

    # Modified invoke to accept TaskSendParams
    def invoke(self, task_send_params: TaskSendParams) -> str:
        """
        Invokes the agent with the given task parameters and returns a final response string.

        This method handles non-streaming execution of the agent's logic based on the
        provided task parameters.

        Args:
            task_send_params (TaskSendParams): Parameters for the task, including
                                               session ID, message content, etc.

        Returns:
            str: A string representation of the agent's final response.
        """
        session = self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=task_send_params.sessionId,
        )
        # Assuming message content is in task_send_params.message
        content = types.Content(
            role='user', parts=task_send_params.message.parts
        )
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=task_send_params.sessionId,
            )
        events = list(
            self._runner.run(
                user_id=self._user_id,
                session_id=session.id,
                new_message=content,
            )
        )
        if not events or not events[-1].content or not events[-1].content.parts:
            return ''
        return '\n'.join([p.text for p in events[-1].content.parts if p.text])

    # Modified stream to accept TaskSendParams
    async def stream(self, task_send_params: TaskSendParams) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response for a task based on the provided parameters.

        This method handles streaming execution, yielding updates and the final response
        as dictionaries.

        Args:
            task_send_params (TaskSendParams): Parameters for the task, including
                                               session ID, message content, etc.

        Yields:
            AsyncIterable[dict[str, Any]]: An asynchronous iterable yielding dictionaries
                                           representing streaming updates and the final response.
                                           Each dictionary contains 'is_task_complete' and
                                           either 'updates' (for streaming) or 'content' (for final).
        """
        session = self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=task_send_params.sessionId,
        )
        # Assuming message content is in task_send_params.message
        content = types.Content(
            role='user', parts=task_send_params.message.parts
        )
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=task_send_params.sessionId,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ''
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = '\n'.join(
                        [p.text for p in event.content.parts if p.text]
                    )
                elif (
                    event.content
                    and event.content.parts
                    and any(
                        [
                            True
                            for p in event.content.parts
                            if p.function_response
                        ]
                    )
                ):
                    response = next(
                        p.function_response.model_dump()
                        for p in event.content.parts
                    )
                yield {
                    'is_task_complete': True,
                    'content': response,
                }
            else:
                yield {
                    'is_task_complete': False,
                    'updates': self.get_processing_message(),
                }


class AgentTaskManager(InMemoryTaskManager):
    """
    A task manager specifically designed to wrap and interact with an AgentWithTaskManager instance.

    This class extends InMemoryTaskManager to integrate an AgentWithTaskManager,
    handling the lifecycle of tasks and managing communication between the agent
    and the task management system.
    """
    def __init__(self, agent: AgentWithTaskManager):
        """
        Initializes the AgentTaskManager with a specific agent instance.

        Args:
            agent (AgentWithTaskManager): The agent instance to be managed.
        """
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """
        Handles the streaming response logic by calling the agent's stream method
        and yielding SendTaskStreamingResponse objects.

        This internal method processes the streaming output from the agent, formats
        it into the appropriate response types, and updates the internal task store.

        Args:
            request (SendTaskStreamingRequest): The incoming streaming task request.

        Yields:
            AsyncIterable[SendTaskStreamingResponse]: An asynchronous iterable yielding
                                                      streaming response objects.

        Returns:
            JSONRPCResponse | None: A JSONRPCResponse error object if an exception occurs
                                    during streaming, otherwise None.
        """
        task_send_params: TaskSendParams = request.params
        # Pass the full task_send_params to the agent's stream method
        try:
            async for item in self.agent.stream(task_send_params):
                is_task_complete = item['is_task_complete']
                artifacts = None
                if not is_task_complete:
                    task_state = TaskState.WORKING
                    parts = [{'type': 'text', 'text': item['updates']}]
                else:
                    if isinstance(item['content'], dict):
                        if (
                            'response' in item['content']
                            and 'result' in item['content']['response']
                        ):
                            data = json.loads(
                                item['content']['response']['result']
                            )
                            task_state = TaskState.INPUT_REQUIRED
                        else:
                            data = item['content']
                            task_state = TaskState.COMPLETED
                        parts = [{'type': 'data', 'data': data}]
                    else:
                        task_state = TaskState.COMPLETED
                        parts = [{'type': 'text', 'text': item['content']}]
                    artifacts = [Artifact(parts=parts, index=0, append=False)]
            message = Message(role='agent', parts=parts)
            task_status = TaskStatus(state=task_state, message=message)
            await self._update_store(
                task_send_params.id, task_status, artifacts
            )
            task_update_event = TaskStatusUpdateEvent(
                id=task_send_params.id,
                status=task_status,
                final=False,
            )
            yield SendTaskStreamingResponse(
                id=request.id, result=task_update_event
            )
            # Now yield Artifacts too
            if artifacts:
                for artifact in artifacts:
                    yield SendTaskStreamingResponse(
                        id=request.id,
                        result=TaskArtifactUpdateEvent(
                            id=task_send_params.id,
                            artifact=artifact,
                        ),
                    )
            if is_task_complete:
                yield SendTaskStreamingResponse(
                    id=request.id,
                    result=TaskStatusUpdateEvent(
                        id=task_send_params.id,
                        status=TaskStatus(
                            state=task_status.state,
                        ),
                        final=True,
                    ),
                )
        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            yield JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message='An error occurred while streaming the response'
                ),
            )

    def _validate_request(
        self, request: SendTaskRequest | SendTaskStreamingRequest
    ) -> None | JSONRPCResponse:
        """
        Validates incoming task requests, specifically checking for compatible output modes.

        Args:
            request (SendTaskRequest | SendTaskStreamingRequest): The incoming task request.

        Returns:
            None | JSONRPCResponse: None if the request is valid, otherwise a JSONRPCResponse
                                    error object indicating incompatibility.
        """
        task_send_params: TaskSendParams = request.params
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes,
            self.agent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                'Unsupported output mode. Received %s, Support %s',
                task_send_params.acceptedOutputModes,
                self.agent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Handles incoming non-streaming task requests.

        This method validates the request, creates or updates the task in the store,
        invokes the agent's non-streaming method, and returns the final task response.

        Args:
            request (SendTaskRequest): The incoming non-streaming task request.

        Returns:
            SendTaskResponse: The final response for the task.
        """
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return await self._invoke(request)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """
        Handles incoming streaming task requests.

        This method validates the request, creates or updates the task in the store,
        and returns an asynchronous iterable for streaming the agent's response.

        Args:
            request (SendTaskStreamingRequest): The incoming streaming task request.

        Returns:
            AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse: An asynchronous
            iterable yielding streaming response objects, or a JSONRPCResponse error
            object if validation fails.
        """
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return self._stream_generator(request)

    async def _update_store(
        self, task_id: str, status: TaskStatus, artifacts: list[Artifact] | None
    ) -> Task:
        """
        Updates the task status and artifacts in the internal store.

        Args:
            task_id (str): The ID of the task to update.
            status (TaskStatus): The new status of the task.
            artifacts (list[Artifact] | None): A list of artifacts to add to the task, or None.

        Returns:
            Task: The updated task object.

        Raises:
            ValueError: If the task with the given ID is not found.
        """
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f'Task {task_id} not found for updating the task')
                raise ValueError(f'Task {task_id} not found')
            task.status = status
            # if status.message is not None:
            #    self.task_messages[task_id].append(status.message)
            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)
            return task

    async def _invoke(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Handles the non-streaming execution by calling the agent's invoke method
        and updating the task state and artifacts.

        Args:
            request (SendTaskRequest): The incoming non-streaming task request.

        Returns:
            SendTaskResponse: The final response for the task.

        Raises:
            ValueError: If an error occurs during agent invocation.
        """
        task_send_params: TaskSendParams = request.params
        # Pass the full task_send_params to the agent's invoke method
        try:
            result = self.agent.invoke(task_send_params)
        except Exception as e:
            logger.error(f'Error invoking agent: {e}')
            raise ValueError(f'Error invoking agent: {e}')
        parts = [{'type': 'text', 'text': result}]
        task_state = (
            TaskState.INPUT_REQUIRED
            if 'MISSING_INFO:' in result
            else TaskState.COMPLETED
        )
        task = await self._update_store(
            task_send_params.id,
            TaskStatus(
                state=task_state, message=Message(role='agent', parts=parts)
            ),
            [Artifact(parts=parts)],
        )
        return SendTaskResponse(id=request.id, result=task)

    # Removed _get_user_query as it's no longer needed
    # def _get_user_query(self, task_send_params: TaskSendParams) -> str:
    #     part = task_send_params.message.parts[0]
    #     if not isinstance(part, TextPart):
    #         raise ValueError('Only text parts are supported')
    #     return part.text
