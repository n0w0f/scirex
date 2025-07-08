from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class TokenUsage:
    """Token usage details from Gemini response."""

    prompt_token_count: int
    candidates_token_count: int
    thoughts_token_count: int
    total_token_count: int
    cached_content_token_count: None | int = None


@dataclass
class GeminiResponse:
    """Complete response object from Gemini API with all important metadata."""

    # Core content
    text: str
    thought_summary: None | str = None

    # Response metadata
    model_version: str = ""
    response_id: None | str = None
    finish_reason: str = ""

    # Token usage (crucial for cost tracking)
    token_usage: None | TokenUsage = None

    # Quality/safety metrics
    safety_ratings: None | list[dict[str, Any]] = None
    avg_logprobs: None | float = None

    # Timing
    timestamp: None | datetime = None

    # Additional metadata
    citation_metadata: None | dict[str, Any] = None
    grounding_metadata: None | dict[str, Any] = None

    @classmethod
    def from_api_response(cls, response) -> "GeminiResponse":
        """Create GeminiResponse from raw Gemini API response."""

        # Extract text and thought content
        text_content = ""
        thought_summary = None

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                for part in candidate.content.parts:
                    if not part.text:
                        continue
                    if hasattr(part, "thought") and part.thought:
                        thought_summary = part.text
                    else:
                        text_content = part.text

        # Extract token usage
        token_usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_token_count=getattr(usage, "prompt_token_count", 0),
                candidates_token_count=getattr(usage, "candidates_token_count", 0),
                thoughts_token_count=getattr(usage, "thoughts_token_count", 0),
                total_token_count=getattr(usage, "total_token_count", 0),
                cached_content_token_count=getattr(usage, "cached_content_token_count", None),
            )

        # Extract other metadata
        model_version = getattr(response, "model_version", "")
        response_id = getattr(response, "response_id", None)

        finish_reason = ""
        safety_ratings = None
        avg_logprobs = None

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            finish_reason = str(getattr(candidate, "finish_reason", ""))
            safety_ratings = getattr(candidate, "safety_ratings", None)
            avg_logprobs = getattr(candidate, "avg_logprobs", None)

        # Convert safety_ratings to serializable format if present
        if safety_ratings:
            try:
                safety_ratings = [
                    asdict(rating) if hasattr(rating, "__dict__") else rating for rating in safety_ratings
                ]
            except Exception:
                safety_ratings = None

        return cls(
            text=text_content,
            thought_summary=thought_summary,
            model_version=model_version,
            response_id=response_id,
            finish_reason=finish_reason,
            token_usage=token_usage,
            safety_ratings=safety_ratings,
            avg_logprobs=avg_logprobs,
            timestamp=datetime.now(),
            citation_metadata=getattr(response, "citation_metadata", None),
            grounding_metadata=getattr(response, "grounding_metadata", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime to ISO string
        if result["timestamp"]:
            result["timestamp"] = result["timestamp"].isoformat()
        return result

    def get_cost_estimate(self, input_cost_per_1k: float = 0.001, output_cost_per_1k: float = 0.002) -> float:
        """Estimate cost based on token usage."""
        if not self.token_usage:
            return 0.0

        prompt_tokens = self.token_usage.prompt_token_count or 0
        candidate_tokens = self.token_usage.candidates_token_count or 0

        input_cost = (prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (candidate_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost
