import unittest


class TrainingHorizonTests(unittest.TestCase):
    def test_training_and_service_use_seventy_two_actions(self):
        from common.config import load_json
        from common.paths import CONFIGS
        from training.worker import policy_horizon_args

        training = load_json(CONFIGS / "training.json")
        service = load_json(CONFIGS / "server.json")

        self.assertEqual(training["chunk_size"], 72)
        self.assertEqual(training["n_action_steps"], 72)
        self.assertEqual(training["steps"], 10000)
        self.assertEqual(training["save_freq"], 1000)
        self.assertEqual(service["actions_per_chunk"], 72)
        self.assertEqual(
            policy_horizon_args(training),
            ["--policy.chunk_size=72", "--policy.n_action_steps=72"],
        )

    def test_policy_horizon_rejects_action_steps_larger_than_chunk(self):
        from training.worker import policy_horizon_args

        with self.assertRaisesRegex(ValueError, "n_action_steps"):
            policy_horizon_args({"chunk_size": 50, "n_action_steps": 72})

    def test_offline_and_runner_defaults_match_service_horizon(self):
        import inspect

        from inference.main import parser
        from service.model_runner import SmolVLARunner

        actions = next(action for action in parser()._actions if action.dest == "actions")
        runner_default = inspect.signature(SmolVLARunner).parameters[
            "actions_per_chunk"
        ].default
        self.assertEqual(actions.default, 72)
        self.assertEqual(runner_default, 72)


if __name__ == "__main__":
    unittest.main()
