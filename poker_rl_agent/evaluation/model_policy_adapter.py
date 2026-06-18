"""Deterministic model-as-opponent adapter for the own-checkpoint ladder.

`ModelPolicyAgent` wraps a loaded ``AlphaHoldemNetwork`` behind the same
``Agent.step(state) -> action`` interface used by everything in
``baseline_agents.py`` (RandomAgent, AlwaysCallAgent, PolicyAgent, ...). This
lets a *second* checkpoint be dropped straight into
``Evaluator.evaluate_duplicate`` as the opponent, so we can run a real,
CI-backed checkpoint-vs-checkpoint ("Stage 0") strength ladder using machinery
that already exists -- rather than leaning on the misleading
"+897 bb/100 vs MCCFR-ES" headline (the MCCFR-ES baseline is effectively
near-random, so that number is NOT a strength signal).

WHY DETERMINISTIC (argmax, not sampling):
    ``evaluate_duplicate`` cancels card luck by replaying the SAME deal twice
    with the agent's seat swapped (antithetic / duplicate poker). That variance
    reduction is only valid if the OPPONENT plays identically across the two
    mirrored seatings of the same deal. A stochastic opponent (Categorical
    sample) would inject fresh RNG into each replay and partially defeat the
    pairing. Taking argmax over the masked logits makes the opponent a pure
    function of the public+private state, so the duplicate-hand cancellation
    actually holds and the bb/100 CI shrinks as intended.

ENCODING/MASKING is intentionally identical to the live evaluator's
``Evaluator._sample_model_action`` (evaluator.py): encode via ``StateEncoder``,
forward through the net, apply ``masked_logits`` to force illegal actions to the
dtype floor, then pick an action. We only change sample() -> argmax(). Encoding
is mirrored here (not imported) to keep this agent self-contained and avoid a
hard dependency on an Evaluator instance, but the StateEncoder + masked_logits
calls are byte-for-byte the same transform.
"""

import torch

from ..environment.state_representation import StateEncoder
from ..models.model_utils import masked_logits
from .baseline_agents import Agent


class ModelPolicyAgent(Agent):
    """A loaded AlphaHoldemNetwork exposed as a deterministic ``Agent`` opponent.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` (AlphaHoldemNetwork) already loaded with checkpoint
        weights and moved to ``device``. Call ``model.eval()`` before passing it.
    num_actions:
        The betting-abstraction action count (FCPA=4, FCHPA=5). MUST match the
        action count of the *agent* this opponent is being evaluated against; a
        mismatch would shape-mismatch inside ``masked_logits`` against the legal
        mask. The ladder CLI guards this before constructing the agent.
    device:
        Device to run inference on. Defaults to the model's own parameter device.
    max_action_history:
        Betting-history truncation length; must match the value the evaluator
        uses (``config.MAX_ACTION_HISTORY``) so the action tower sees the same
        sequence length the model was trained/encoded with.
    """

    def __init__(self, model, num_actions, device=None, max_action_history=64):
        self.model = model
        self.num_actions = int(num_actions)
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        # Encoder builds tensors on CPU then we move the batch to the model
        # device, exactly like Evaluator._sample_model_action does.
        self.encoder = StateEncoder(device="cpu", max_action_history=int(max_action_history))

    @torch.no_grad()
    def step(self, state):
        """Return the deterministic (argmax) legal action for ``state``.

        Mirrors ``Evaluator._sample_model_action`` but replaces the Categorical
        sample with an argmax over the masked logits so the opponent is a pure
        function of the state (see module docstring on why this matters for the
        duplicate-hand CI).
        """
        player_id = state.current_player()
        encoded = self.encoder.encode_state(state, player_id, self.num_actions)
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}

        outputs = self.model(batch)
        logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
        # argmax over masked logits: illegal actions are at the dtype floor, so
        # the chosen action is always legal. Deterministic given the state.
        return int(torch.argmax(logits, dim=-1).item())
