"""Single-turn SkyRL environment; the model sees only the published prompt."""

from pathlib import Path

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ground_truth_from_extras
from skyrl_gym.verification import VerificationResult

from .scoring import load_scorer, score, tool_events

SCORER = load_scorer(Path(__file__).parent / "native_verifier")


class PDBThinkEnv(BaseTextEnv):
    def __init__(self, env_config, extras):
        super().__init__()
        self.ground_truth = ground_truth_from_extras(extras)
        self.family = extras["family"]
        self.path = extras["path"]
        self.evidence = None
        self.metrics = {}

    def init(self, prompt):
        if [m["role"] for m in prompt] != ["system", "user"]:
            raise ValueError("PDBThink requires exactly the original system/user messages")
        return prompt, {
            "chat_completion_params": {
                "tools": [],
                "tool_choice": "none",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        }

    def set_rollout_evidence(self, evidence):
        self.evidence = evidence

    def step(self, action):
        if self.evidence is None:
            raise RuntimeError("Missing generation evidence; cannot verify termination or tool use")
        message = self.evidence.metadata.get("assistant_message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise RuntimeError("Missing original structured assistant message")
        # SkyRL's action text can strip thinking blocks. The native benchmark
        # and held-out evaluator score the original API content, so use that
        # exact content here too, without a second answer-extraction policy.
        result = score(
            SCORER,
            message.get("content") or "",
            self.ground_truth,
            truncated=self.evidence.stop_reason == "length",
            refusal=bool(message.get("refusal")),
            tool_violation=tool_events(self.evidence.messages) or tool_events(self.evidence.metadata),
        )
        self.metrics = {
            k: float(result[k]) for k in ("reward", "format_error", "truncated", "refusal", "tool_violation")
        }
        self.metrics["output_tokens"] = self.evidence.generated_token_count
        return BaseTextEnvStepOutput(
            observations=[],
            reward=result["reward"],
            done=True,
            metadata=self.metrics,
            verification=VerificationResult.verified(
                result["reward"],
                passed=bool(result["reward"]),
                diagnostics={"task": self.path, "family": self.family, **result},
            ),
        )

    def get_metrics(self):
        return self.metrics
