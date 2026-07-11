import torch

from trajectory_extractor.steering import SteeringHook, normalize_direction


class AddOneLayer(torch.nn.Module):
    def forward(self, hidden_states):
        return (hidden_states + 1.0,)


class FakeLlama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([AddOneLayer(), AddOneLayer()])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def test_normalize_direction_returns_unit_vector():
    direction = normalize_direction(torch.tensor([3.0, 4.0]))
    assert torch.linalg.vector_norm(direction).item() == 1.0


def test_steering_hook_changes_last_token_and_restores_hook():
    model = FakeLlama()
    hidden = torch.zeros((1, 2, 2))
    baseline = model(hidden)

    with SteeringHook(model, layer_idx=0, direction=torch.tensor([1.0, 0.0]), strength=2.0):
        steered = model(hidden)

    restored = model(hidden)
    torch.testing.assert_close(steered[0, -1], baseline[0, -1] + torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(restored, baseline)
