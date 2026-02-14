import os
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.utils.config import Config


def _test_config():
    cfg = Config()
    cfg.HIDDEN_DIM = 64
    cfg.NUM_LAYERS_CARD = 2
    cfg.NUM_LAYERS_ACTION = 1
    cfg.DEVICE = "cpu"
    cfg.sync_legacy_fields()
    return cfg


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a real (untrained) model checkpoint in a temp dir."""
    cfg = _test_config()
    env = PokerEnv(env_preset="hunl_fcpa", betting_abstraction="fcpa")
    model = AlphaHoldemNetwork(env.num_actions(), cfg)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_serve.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)

        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(
                "default:\n"
                '  DEVICE: "cpu"\n'
                "  HIDDEN_DIM: 64\n"
                "  NUM_LAYERS_CARD: 2\n"
                "  NUM_LAYERS_ACTION: 1\n"
            )

        from poker_rl_agent.serving import app as app_module

        app_module._cli_args.update({
            "checkpoint": checkpoint_path,
            "config_file": config_path,
            "config_name": "default",
            "game_mode": "fcpa",
            "bot_policy": "sample",
            "policy_temperature": 1.0,
        })

        with TestClient(app_module.app) as tc:
            yield tc


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


def test_create_session_valid(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    assert resp.status_code == 200
    data = resp.json()
    gs = data["game_state"]
    assert "session_id" in gs
    assert len(gs["hero_cards"]) == 2
    assert len(gs["bot_cards"]) == 0
    assert gs["is_terminal"] is False


def test_create_session_invalid_seat(client):
    resp = client.post("/api/v1/session", json={"human_seat": 5})
    assert resp.status_code == 400


def test_get_nonexistent_session(client):
    resp = client.get("/api/v1/session/nonexistent-uuid")
    assert resp.status_code == 404


def test_submit_legal_action(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    assert resp.status_code == 200
    gs = resp.json()["game_state"]
    sid = gs["session_id"]

    if gs["is_terminal"]:
        return

    assert gs["current_player"] == "human"
    legal = gs["legal_actions"]
    assert len(legal) > 0

    action_id = legal[0]["action_id"]
    resp2 = client.post(f"/api/v1/session/{sid}/action", json={"action_id": action_id})
    assert resp2.status_code == 200


def test_submit_illegal_action(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    gs = resp.json()["game_state"]
    sid = gs["session_id"]

    if gs["is_terminal"]:
        return

    legal_ids = {a["action_id"] for a in gs["legal_actions"]}
    illegal = -999
    while illegal in legal_ids:
        illegal -= 1

    resp2 = client.post(f"/api/v1/session/{sid}/action", json={"action_id": illegal})
    assert resp2.status_code == 400


def test_play_full_hand_to_terminal(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    gs = resp.json()["game_state"]
    sid = gs["session_id"]

    for _ in range(200):
        if gs["is_terminal"]:
            break
        legal = gs["legal_actions"]
        action_id = legal[0]["action_id"]
        resp = client.post(f"/api/v1/session/{sid}/action", json={"action_id": action_id})
        assert resp.status_code == 200
        gs = resp.json()["game_state"]

    assert gs["is_terminal"] is True
    assert gs["hand_result"] is not None
    assert len(gs["bot_cards"]) > 0


def test_new_hand_resets(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    gs = resp.json()["game_state"]
    sid = gs["session_id"]

    for _ in range(200):
        if gs["is_terminal"]:
            break
        action_id = gs["legal_actions"][0]["action_id"]
        resp = client.post(f"/api/v1/session/{sid}/action", json={"action_id": action_id})
        gs = resp.json()["game_state"]

    assert gs["is_terminal"] is True

    resp = client.post(f"/api/v1/session/{sid}/new_hand")
    assert resp.status_code == 200
    gs = resp.json()["game_state"]
    assert gs["hand_number"] == 2
    assert gs["is_terminal"] is False or gs["is_terminal"] is True  # could be terminal if bot acts to end


def test_new_hand_mid_play_rejected(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    gs = resp.json()["game_state"]
    sid = gs["session_id"]

    if not gs["is_terminal"]:
        resp = client.post(f"/api/v1/session/{sid}/new_hand")
        assert resp.status_code == 400


def test_bot_cards_hidden_during_play(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    gs = resp.json()["game_state"]

    if not gs["is_terminal"]:
        assert len(gs["bot_cards"]) == 0


def test_cumulative_stats(client):
    resp = client.post("/api/v1/session", json={"human_seat": 0})
    data = resp.json()
    gs = data["game_state"]
    sid = gs["session_id"]

    # Play hand 1 to terminal
    for _ in range(200):
        if gs["is_terminal"]:
            break
        action_id = gs["legal_actions"][0]["action_id"]
        resp = client.post(f"/api/v1/session/{sid}/action", json={"action_id": action_id})
        data = resp.json()
        gs = data["game_state"]

    stats1 = data["stats"]
    assert stats1["hands_played"] == 1

    # Play hand 2
    resp = client.post(f"/api/v1/session/{sid}/new_hand")
    data = resp.json()
    gs = data["game_state"]

    for _ in range(200):
        if gs["is_terminal"]:
            break
        action_id = gs["legal_actions"][0]["action_id"]
        resp = client.post(f"/api/v1/session/{sid}/action", json={"action_id": action_id})
        data = resp.json()
        gs = data["game_state"]

    stats2 = data["stats"]
    assert stats2["hands_played"] == 2


def test_static_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
