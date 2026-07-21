import unittest

import torch

from stego.router import EnergyPreservingHaarBandRouter


class EnergyPreservingHaarBandRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    @staticmethod
    def _make_nonidentity(router: EnergyPreservingHaarBandRouter) -> None:
        final_layer = router.skew_predictor[-1]
        with torch.no_grad():
            final_layer.weight.normal_(mean=0.0, std=0.15)
            final_layer.bias.copy_(torch.linspace(-0.2, 0.2, 6))

    def test_identity_initialization_preserves_values_exactly(self) -> None:
        router = EnergyPreservingHaarBandRouter()
        values = torch.randn(3, 12, 9, 7)

        routed = router(values)
        matrices = router.routing_matrix(values)
        identity = torch.eye(4).expand(3, -1, -1)

        self.assertTrue(torch.equal(routed, values))
        self.assertTrue(torch.equal(matrices, identity))

    def test_cayley_matrices_are_orthogonal(self) -> None:
        router = EnergyPreservingHaarBandRouter()
        self._make_nonidentity(router)
        values = torch.randn(5, 20, 6, 8)

        matrices = router.routing_matrix(values)
        products = matrices.transpose(-1, -2) @ matrices
        identity = torch.eye(4).expand_as(products)

        self.assertTrue(torch.allclose(products, identity, atol=2e-6, rtol=2e-6))

    def test_nonidentity_routing_preserves_l2_energy(self) -> None:
        router = EnergyPreservingHaarBandRouter()
        self._make_nonidentity(router)
        values = torch.randn(4, 12, 11, 13)

        routed = router(values)
        input_energy = values.square().sum(dim=(1, 2, 3))
        output_energy = routed.square().sum(dim=(1, 2, 3))

        self.assertFalse(torch.allclose(routed, values))
        self.assertTrue(torch.allclose(output_energy, input_energy, atol=2e-4, rtol=2e-6))

    def test_gradients_reach_inputs_and_predictor(self) -> None:
        router = EnergyPreservingHaarBandRouter()
        self._make_nonidentity(router)
        values = torch.randn(2, 12, 5, 7, requires_grad=True)
        target = torch.randn_like(values)

        loss = (router(values) * target).mean()
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().sum().item(), 0.0)
        for parameter in router.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_zero_energy_input_has_finite_gradients(self) -> None:
        router = EnergyPreservingHaarBandRouter()
        values = torch.zeros(2, 12, 5, 7, requires_grad=True)
        target = torch.randn_like(values)

        loss = (router(values) * target).sum()
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        for parameter in router.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
