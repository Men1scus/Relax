# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Argument parsing and validation must read the registry, not string lists."""

import argparse
import importlib
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


ARGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


@pytest.fixture()
def arguments_module(monkeypatch):
    """Import relax.utils.arguments with its heavy optional deps stubbed out.

    Mirrors tests/utils/test_arguments_opd_teacher_colocate.py.
    """
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


def _args(estimator="gdpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        normalize_advantages=False,
        rewards_normalization=True,
        custom_reward_post_process_path=None,
        n_samples_per_prompt=4,
        gdpo_reward_keys=["correctness", "format"],
        gdpo_reward_weights=None,
        reward_key="score",
        use_critic=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------- source-level: no hardcoded name lists ----------------


def test_choices_come_from_the_registry():
    assert "choices=list_algorithm_names()" in ARGS_PATH.read_text(encoding="utf-8")


def test_no_hardcoded_estimator_choice_list():
    src = ARGS_PATH.read_text(encoding="utf-8")
    assert '"reinforce_plus_plus_baseline",\n                    "ppo",' not in src


def test_no_estimator_name_comparisons_remain():
    src = ARGS_PATH.read_text(encoding="utf-8")
    for banned in (
        'args.advantage_estimator == "ppo"',
        'args.advantage_estimator in ["reinforce_plus_plus"',
    ):
        assert banned not in src, f"arguments.py still contains: {banned}"


def test_validation_reads_spec_fields():
    src = ARGS_PATH.read_text(encoding="utf-8")
    for field in (
        "needs_critic",
        "requires_normalize_advantages",
        "forbids_normalize_advantages",
        "requires_rewards_normalization",
        "allows_custom_reward_post_process",
        "min_group_size",
        "uses_reward_components",
        "disabled_reason",
    ):
        assert field in src, f"arguments.py does not consult spec.{field}"


def test_gdpo_arguments_declared():
    src = ARGS_PATH.read_text(encoding="utf-8")
    assert "--gdpo-reward-keys" in src
    assert "--gdpo-reward-weights" in src


# ---------------- behaviour ----------------


def test_parser_accepts_gdpo_and_its_options(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(
        [
            "--advantage-estimator",
            "gdpo",
            "--gdpo-reward-keys",
            "correctness",
            "format",
            "--gdpo-reward-weights",
            "1.0",
            "0.5",
        ]
    )

    assert args.advantage_estimator == "gdpo"
    assert args.gdpo_reward_keys == ["correctness", "format"]
    assert args.gdpo_reward_weights == [1.0, 0.5]


def test_parser_rejects_an_unregistered_estimator(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--advantage-estimator", "not_an_algorithm"])


def test_every_registered_algorithm_is_an_accepted_choice(arguments_module):
    from relax.algorithms import list_algorithm_names

    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    for name in list_algorithm_names():
        parsed = parser.parse_args(["--advantage-estimator", name])
        assert parsed.advantage_estimator == name


def test_gdpo_rejects_custom_reward_post_process(arguments_module):
    with pytest.raises(ValueError, match="custom-reward-post-process-path"):
        arguments_module.validate_algorithm_args(_args(custom_reward_post_process_path="pkg.mod.fn"))


def test_gdpo_rejects_normalize_advantages(arguments_module):
    with pytest.raises(ValueError, match="normalize-advantages"):
        arguments_module.validate_algorithm_args(_args(normalize_advantages=True))


def test_gdpo_rejects_group_size_below_two(arguments_module):
    with pytest.raises(ValueError, match="n-samples-per-prompt"):
        arguments_module.validate_algorithm_args(_args(n_samples_per_prompt=1))


def test_gdpo_rejects_disabled_rewards_normalization(arguments_module):
    with pytest.raises(ValueError, match="reward normalization"):
        arguments_module.validate_algorithm_args(_args(rewards_normalization=False))


def test_gdpo_requires_at_least_two_keys(arguments_module):
    with pytest.raises(ValueError, match="at least two reward keys"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_keys=["correctness"]))


def test_gdpo_rejects_duplicate_keys(arguments_module):
    with pytest.raises(ValueError, match="duplicates"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_keys=["a", "a"]))


def test_gdpo_rejects_weight_count_mismatch(arguments_module):
    with pytest.raises(ValueError, match="gdpo-reward-weights"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_weights=[1.0]))


def test_gdpo_requires_reward_key(arguments_module):
    with pytest.raises(ValueError, match="reward-key"):
        arguments_module.validate_algorithm_args(_args(reward_key=None))


def test_gdpo_happy_path(arguments_module):
    args = _args()
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_reinforce_family_requires_normalize_advantages(arguments_module):
    for estimator in ("reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        with pytest.raises(ValueError, match="normalize-advantages"):
            arguments_module.validate_algorithm_args(
                _args(estimator, normalize_advantages=False, gdpo_reward_keys=None)
            )


def test_reinforce_family_passes_with_normalize_advantages(arguments_module):
    args = _args("reinforce_plus_plus", normalize_advantages=True, gdpo_reward_keys=None)
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_ppo_is_rejected_with_its_disabled_reason(arguments_module):
    with pytest.raises(ValueError, match="no longer supported"):
        arguments_module.validate_algorithm_args(_args("ppo", gdpo_reward_keys=None))


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_passes_with_defaults(arguments_module, estimator):
    args = _args(estimator, gdpo_reward_keys=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_non_gdpo_algorithms_ignore_reward_key_and_component_options(arguments_module):
    """Only multi-reward algorithms police --reward-key / --gdpo-*."""
    args = _args("grpo", gdpo_reward_keys=None, gdpo_reward_weights=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)


def test_unknown_estimator_raises_from_the_registry(arguments_module):
    with pytest.raises(KeyError, match="Unknown advantage estimator"):
        arguments_module.validate_algorithm_args(_args("not_an_algorithm"))


# ---------------- --custom-config-path override timing ----------------


def _write_yaml(tmp_path, body):
    path = tmp_path / "override.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _overridable_args(tmp_path, body, **overrides):
    """Args as they look when the YAML merge runs: already validated once."""
    base = _args("grpo", gdpo_reward_keys=None, reward_key=None)
    base.loss_type = "policy_loss"
    base.custom_config_path = _write_yaml(tmp_path, body)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_yaml_cannot_smuggle_in_a_disabled_estimator(arguments_module, tmp_path):
    """main rejected `advantage_estimator: ppo` from YAML; so must we."""
    args = _overridable_args(tmp_path, "advantage_estimator: ppo\n")

    with pytest.raises(ValueError, match="no longer supported"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_cannot_bypass_gdpo_requirements(arguments_module, tmp_path):
    """Switching to GDPO from YAML must still demand its reward keys."""
    args = _overridable_args(tmp_path, "advantage_estimator: gdpo\n")

    with pytest.raises(ValueError, match="at least two reward keys"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_cannot_enable_conflicting_whitening_under_gdpo(arguments_module, tmp_path):
    """The dangerous case: silent double whitening rather than a crash."""
    args = _overridable_args(
        tmp_path,
        "normalize_advantages: true\n",
        advantage_estimator="gdpo",
        gdpo_reward_keys=["correctness", "format"],
        reward_key="score",
        n_samples_per_prompt=8,
    )

    with pytest.raises(ValueError, match="normalize-advantages"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_without_algorithm_changes_is_accepted(arguments_module, tmp_path):
    args = _overridable_args(tmp_path, "lr: 0.5\n")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5
    assert args.advantage_estimator == "grpo"


def test_no_yaml_is_a_no_op(arguments_module):
    args = _args("grpo", gdpo_reward_keys=None, reward_key=None)
    args.loss_type = "policy_loss"
    args.custom_config_path = None
    arguments_module.apply_custom_config_overrides(args)


def test_sft_runs_skip_the_algorithm_recheck(arguments_module, tmp_path):
    """SFT never selects an estimator, so a stale one must not block it."""
    args = _overridable_args(tmp_path, "lr: 0.5\n", loss_type="sft", advantage_estimator="ppo")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5


def test_slime_validate_args_applies_overrides_through_the_helper(arguments_module):
    """Guard the call site: the merge must go through the re-checking
    helper."""
    import inspect

    src = inspect.getsource(arguments_module.slime_validate_args)
    assert "apply_custom_config_overrides(args)" in src
    assert "yaml.safe_load" not in src, "the YAML merge was inlined again, skipping the re-check"


def test_spec_with_an_unregistered_implementation_is_rejected_at_startup(arguments_module, monkeypatch):
    """A registry typo must name itself, not KeyError inside a worker."""
    from dataclasses import replace

    from relax.algorithms.spec import ALGORITHM_SPECS

    broken = replace(ALGORITHM_SPECS["grpo"], advantage_fn="typo_does_not_exist")
    monkeypatch.setitem(ALGORITHM_SPECS, "grpo", broken)

    with pytest.raises(ValueError, match="typo_does_not_exist"):
        arguments_module.validate_algorithm_args(_args("grpo", gdpo_reward_keys=None, reward_key=None))
