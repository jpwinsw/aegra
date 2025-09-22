"""
Enhanced Langfuse integration for Aegra with advanced features.
Provides scoring, datasets, and improved metadata tracking.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

_LANGFUSE_LOGGING_ENABLED = os.getenv("LANGFUSE_LOGGING", "false").lower() == "true"
_langfuse_client = None
_langfuse_handler = None


class LangfuseEnhanced:
    """Enhanced Langfuse integration with advanced features."""

    def __init__(self):
        self.enabled = _LANGFUSE_LOGGING_ENABLED
        self.client = None
        self.handler = None

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize Langfuse client and handler."""
        global _langfuse_client, _langfuse_handler

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            # Validate required environment variables
            required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            if missing_vars:
                logger.warning(
                    f"LANGFUSE_LOGGING is enabled but missing required environment variables: {missing_vars}. "
                    "Please set these in your .env file."
                )
                self.enabled = False
                return

            # Initialize or reuse global client
            if _langfuse_client is None:
                _langfuse_client = get_client()

                # Verify connection
                if not _langfuse_client.auth_check():
                    logger.warning(
                        "Failed to authenticate with Langfuse. Please check your credentials and host."
                    )
                    self.enabled = False
                    return

                logger.info("Langfuse client authenticated successfully")

            # Initialize or reuse global handler
            if _langfuse_handler is None:
                _langfuse_handler = CallbackHandler()
                logger.info("Langfuse CallbackHandler initialized")

            self.client = _langfuse_client
            self.handler = _langfuse_handler

        except ImportError:
            logger.warning(
                "LANGFUSE_LOGGING is true, but 'langfuse' is not installed. "
                "Please run 'pip install langfuse' to enable tracing."
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False

    def get_callbacks(self, metadata: Optional[Dict[str, Any]] = None) -> List:
        """
        Get callbacks with optional metadata for LangGraph execution.

        Args:
            metadata: Optional metadata to include in traces

        Returns:
            List of callbacks for LangGraph/LangChain
        """
        if not self.enabled or not self.handler:
            return []

        # Note: Metadata will be passed through config in actual execution
        return [self.handler]

    @contextmanager
    def trace_agent_run(
        self,
        agent_name: str,
        thread_id: str,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing an agent run with rich metadata.

        Args:
            agent_name: Name of the agent being executed
            thread_id: Thread ID for the conversation
            run_id: Optional run ID for trace correlation
            user_id: Optional user ID for filtering
            metadata: Additional metadata to include
        """
        if not self.enabled or not self.client:
            yield None
            return

        try:
            # Build comprehensive tags
            tags = [
                "aegra_run",
                f"agent:{agent_name}",
                f"thread:{thread_id}"
            ]

            if run_id:
                tags.append(f"run:{run_id}")

            if user_id:
                tags.append(f"user:{user_id}")

            # Add timestamp to metadata
            enhanced_metadata = {
                "agent_name": agent_name,
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }

            # Create span with enhanced metadata
            with self.client.start_as_current_span(
                name=f"{agent_name}-run",
                session_id=thread_id,
                user_id=user_id,
                metadata=enhanced_metadata,
                tags=tags,
                trace_id=run_id  # Use run_id as trace_id for correlation
            ) as span:
                yield span

        except Exception as e:
            logger.error(f"Error in Langfuse trace context: {e}")
            yield None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        data_type: str = "NUMERIC",
        comment: Optional[str] = None
    ):
        """
        Score a trace for evaluation.

        Args:
            trace_id: ID of the trace to score
            name: Name of the score metric
            value: Score value
            data_type: Type of score (NUMERIC, BOOLEAN, CATEGORICAL)
            comment: Optional comment about the score
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment
            )
            logger.debug(f"Created score '{name}' for trace {trace_id}")
        except Exception as e:
            logger.error(f"Error scoring trace: {e}")

    def log_user_feedback(
        self,
        trace_id: str,
        feedback_value: int,
        comment: Optional[str] = None
    ):
        """
        Log user feedback for a trace.

        Args:
            trace_id: ID of the trace
            feedback_value: Feedback value (1 for positive, 0 for negative)
            comment: Optional feedback comment
        """
        self.score_trace(
            trace_id=trace_id,
            name="user-feedback",
            value=feedback_value,
            data_type="NUMERIC",
            comment=comment
        )

    def log_llm_judge_score(
        self,
        trace_id: str,
        evaluation_name: str,
        score: float,
        reasoning: Optional[str] = None
    ):
        """
        Log LLM-as-a-Judge evaluation score.

        Args:
            trace_id: ID of the trace
            evaluation_name: Name of evaluation (e.g., "toxicity", "correctness")
            score: Evaluation score
            reasoning: Optional reasoning for the score
        """
        self.score_trace(
            trace_id=trace_id,
            name=f"llm-judge-{evaluation_name}",
            value=score,
            data_type="NUMERIC",
            comment=reasoning
        )

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a dataset in Langfuse for offline evaluation.

        Args:
            name: Name of the dataset
            description: Optional description
            metadata: Optional metadata dictionary

        Returns:
            Dataset object or None if failed
        """
        if not self.enabled or not self.client:
            return None

        try:
            dataset = self.client.create_dataset(
                name=name,
                description=description,
                metadata={
                    "created_by": "aegra",
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            logger.info(f"Created Langfuse dataset: {name}")
            return dataset
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return None

    def add_dataset_item(
        self,
        dataset_name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an item to a Langfuse dataset.

        Args:
            dataset_name: Name of the dataset
            input_data: Input data for the item
            expected_output: Expected output for evaluation
            metadata: Optional metadata

        Returns:
            Dataset item or None if failed
        """
        if not self.enabled or not self.client:
            return None

        try:
            item = self.client.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata
            )
            logger.debug(f"Added item to dataset: {dataset_name}")
            return item
        except Exception as e:
            logger.error(f"Error adding dataset item: {e}")
            return None

    def run_on_dataset(
        self,
        dataset_name: str,
        run_name: str,
        agent_executor,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Run an agent on a dataset for batch evaluation.

        Args:
            dataset_name: Name of the dataset to run on
            run_name: Name for this evaluation run
            agent_executor: The agent/graph to execute
            config: Optional configuration for the agent
        """
        if not self.enabled or not self.client:
            logger.warning("Langfuse not enabled, skipping dataset run")
            return []

        try:
            dataset = self.client.get_dataset(dataset_name)
            results = []

            for item in dataset.items:
                # Create a trace for this dataset item run
                with item.run(
                    run_name=run_name,
                    run_metadata={
                        "agent_type": config.get("agent_name", "unknown"),
                        "model": config.get("model", "unknown")
                    }
                ) as span:
                    try:
                        # Execute the agent with the input
                        output = agent_executor(
                            item.input,
                            config={
                                **(config or {}),
                                "callbacks": self.get_callbacks()
                            }
                        )

                        # Update the span with output
                        span.update(output=output)

                        # Optionally score against expected output
                        if item.expected_output:
                            # Here you could add automatic scoring logic
                            pass

                        results.append({
                            "input": item.input,
                            "output": output,
                            "expected": item.expected_output
                        })

                    except Exception as e:
                        logger.error(f"Error running item {item.id}: {e}")
                        span.update(output={"error": str(e)})
                        results.append({
                            "input": item.input,
                            "error": str(e)
                        })

            logger.info(f"Completed dataset run '{run_name}' on {len(results)} items")
            return results

        except Exception as e:
            logger.error(f"Error running on dataset: {e}")
            return []

    def flush(self):
        """Flush any pending data to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.debug("Flushed Langfuse data")
            except Exception as e:
                logger.error(f"Error flushing Langfuse data: {e}")


# Global singleton instance
_enhanced_instance = None


def get_enhanced_langfuse() -> LangfuseEnhanced:
    """Get or create the global enhanced Langfuse instance."""
    global _enhanced_instance
    if _enhanced_instance is None:
        _enhanced_instance = LangfuseEnhanced()
    return _enhanced_instance


# Convenience functions for backward compatibility
def get_tracing_callbacks(metadata: Optional[Dict[str, Any]] = None) -> List:
    """
    Get tracing callbacks (backward compatible with existing code).

    Args:
        metadata: Optional metadata to include

    Returns:
        List of callbacks for LangGraph execution
    """
    instance = get_enhanced_langfuse()
    return instance.get_callbacks(metadata)


@contextmanager
def trace_agent_run(
    agent_name: str,
    thread_id: str,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience context manager for tracing agent runs.

    Example:
        with trace_agent_run("weather_agent", thread_id, run_id, user_id) as span:
            result = await graph.ainvoke(input_data, config)
            if span:
                span.update(output=result)
    """
    instance = get_enhanced_langfuse()
    with instance.trace_agent_run(agent_name, thread_id, run_id, user_id, metadata) as span:
        yield span