import torch

from model import MappoAgent
from ppo_utils import load_matching_weights


def test_load_matching_weights_transfer(tmp_path):
    device = torch.device("cpu")

    # Create two MappoAgents with different state dimensions (Stage 0 vs Stage 1)
    agent_stage0 = MappoAgent(state_dim=135).to(device)
    agent_stage1 = MappoAgent(state_dim=243).to(device)

    # Save stage 0 checkpoint
    chk_dir = tmp_path / "mappo_s0"
    chk_dir.mkdir()
    chk_file = chk_dir / "policy.pt"
    torch.save(agent_stage0.state_dict(), chk_file)

    # Assert actor weights are initially different between stage 1 and stage 0
    # Modifying stage 0 actor weights to make them distinct
    with torch.no_grad():
        for param in agent_stage0.actor.parameters():
            param.fill_(0.5)

    torch.save(agent_stage0.state_dict(), chk_file)

    # Transfer matching weights from stage 0 checkpoint to stage 1 agent
    success = load_matching_weights(agent_stage1, str(chk_file), device)

    assert success is True

    # Verify that actor weights transferred successfully
    for param in agent_stage1.actor.parameters():
        assert torch.all(param == 0.5)

    # Verify that critic weights did not transfer (since they had shape mismatch and were reinitialized/kept original)
    assert not torch.all(agent_stage1.critic[0].weight == 0.5)
