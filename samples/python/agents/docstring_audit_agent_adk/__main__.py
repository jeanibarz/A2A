import logging
import os

import click

from agent import DocstringAuditADKAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from dotenv import load_dotenv
from task_manager import AgentTaskManager


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10002)
def main(host, port):
    try:
        # Check for API key only if Vertex AI is not configured
        if not os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
                )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='audit_docstrings',
            name='Audit Docstrings Tool',
            description='Audits Python code for docstring consistency using an LLM.',
            tags=['docstring', 'audit', 'python'],
            examples=[
                'Audit the docstrings in these files.',
                'Check docstrings for file1.py and file2.py',
            ],
        )
        agent_card = AgentCard(
            name='Docstring Audit ADK Agent',
            description='This agent audits Python code for docstring consistency using an LLM and the ADK framework.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=DocstringAuditADKAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DocstringAuditADKAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=DocstringAuditADKAgent()),
            host=host,
            port=port,
        )
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
