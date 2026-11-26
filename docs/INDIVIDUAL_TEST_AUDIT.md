# Individual investigation of previous failed, errored and skipped tests

All 29 failed cases, 12 setup errors and the one CUDA skip are listed individually below.
Classification comes from each JUnit traceback plus its test source. A classified cause is
not a passed rerun. No organizer test or validator was changed.

## 1. `tests/test_env.py::test_observation_matches_space`

Previous result: **failure**. Source line: 32.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_observation_matches_space' -q -ra
```

## 2. `tests/test_env.py::test_rays_walls_and_doors`

Previous result: **failure**. Source line: 43.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_rays_walls_and_doors' -q -ra
```

## 3. `tests/test_env.py::test_locked_door_ray_type`

Previous result: **failure**. Source line: 56.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_locked_door_ray_type' -q -ra
```

## 4. `tests/test_env.py::test_enemy_and_item_rays_and_entities`

Previous result: **failure**. Source line: 63.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_enemy_and_item_rays_and_entities' -q -ra
```

## 5. `tests/test_env.py::test_line_of_sight_and_fov`

Previous result: **failure**. Source line: 78.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_line_of_sight_and_fov' -q -ra
```

## 6. `tests/test_env.py::test_key_pickup_and_locked_door_exit`

Previous result: **failure**. Source line: 88.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_key_pickup_and_locked_door_exit' -q -ra
```

## 7. `tests/test_env.py::test_truncation_at_tick_cap`

Previous result: **failure**. Source line: 108.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_truncation_at_tick_cap' -q -ra
```

## 8. `tests/test_env.py::test_translation_invariance`

Previous result: **failure**. Source line: 122.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_translation_invariance' -q -ra
```

## 9. `tests/test_env.py::test_invalid_actions_rejected`

Previous result: **failure**. Source line: 144.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_invalid_actions_rejected' -q -ra
```

## 10. `tests/test_env.py::test_freedoom_map_runs`

Previous result: **failure**. Source line: 154.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_freedoom_map_runs' -q -ra
```

## 11. `tests/test_env.py::test_projectiles_are_classified`

Previous result: **failure**. Source line: 169.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_projectiles_are_classified' -q -ra
```

## 12. `tests/test_env.py::test_final_step_variables_are_refreshed`

Previous result: **failure**. Source line: 184.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_final_step_variables_are_refreshed' -q -ra
```

## 13. `tests/test_env.py::test_static_geometry_comes_from_the_wad`

Previous result: **failure**. Source line: 197.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_static_geometry_comes_from_the_wad' -q -ra
```

## 14. `tests/test_env.py::test_out_of_order_calls_raise_clear_errors`

Previous result: **failure**. Source line: 208.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_env.py::test_out_of_order_calls_raise_clear_errors' -q -ra
```

## 15. `tests/test_nexus.py::test_generated_curriculum_engine_loads`

Previous result: **failure**. Source line: 114.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_nexus.py::test_generated_curriculum_engine_loads' -q -ra
```

## 16. `tests/test_nexus.py::test_checkpoint_load_once_and_readonly_act`

Previous result: **failure**. Source line: 146.

Cause: DoomAgent cannot initialize because genuine trained weights do not exist.

Action: Complete real PPO updates, validate and promote a checkpoint, then rerun this case.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_nexus.py::test_checkpoint_load_once_and_readonly_act' -q -ra
```

## 17. `tests/test_nexus.py::test_memory_reset_and_independent_instances`

Previous result: **failure**. Source line: 160.

Cause: DoomAgent cannot initialize because genuine trained weights do not exist.

Action: Complete real PPO updates, validate and promote a checkpoint, then rerun this case.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_nexus.py::test_memory_reset_and_independent_instances' -q -ra
```

## 18. `tests/test_nexus.py::test_real_episode_without_invalid_actions`

Previous result: **failure**. Source line: 172.

Cause: DoomAgent cannot initialize because genuine trained weights do not exist.

Action: Complete real PPO updates, validate and promote a checkpoint, then rerun this case.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_nexus.py::test_real_episode_without_invalid_actions' -q -ra
```

## 19. `tests/test_nexus.py::test_cuda_agent`

Previous result: **skipped**. Source line: 185.

Cause: CPU-only PyTorch; RTX 2050 hardware was detected, but CUDA agent execution was not tested.

Action: Install CUDA PyTorch in WSL 2; pass CUDA tensor preflight, train real weights, rerun this case.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_nexus.py::test_cuda_agent' -q -ra
```

## 20. `tests/test_remote.py::test_good_agent_full_episode`

Previous result: **error**. Source line: 81.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_good_agent_full_episode' -q -ra
```

## 21. `tests/test_remote.py::test_loopback_tcp_fallback`

Previous result: **error**. Source line: 97.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_loopback_tcp_fallback' -q -ra
```

## 22. `tests/test_remote.py::test_misbehaving_agents[init_raises-init-error]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[init_raises-init-error]' -q -ra
```

## 23. `tests/test_remote.py::test_misbehaving_agents[sys_exit-init-error]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[sys_exit-init-error]' -q -ra
```

## 24. `tests/test_remote.py::test_misbehaving_agents[no_class-init-error]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[no_class-init-error]' -q -ra
```

## 25. `tests/test_remote.py::test_misbehaving_agents[act_raises-act-error]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[act_raises-act-error]' -q -ra
```

## 26. `tests/test_remote.py::test_misbehaving_agents[act_hangs-act-timeout]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[act_hangs-act-timeout]' -q -ra
```

## 27. `tests/test_remote.py::test_misbehaving_agents[bad_shape-act-invalid_action]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[bad_shape-act-invalid_action]' -q -ra
```

## 28. `tests/test_remote.py::test_misbehaving_agents[bad_values-act-invalid_action]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[bad_values-act-invalid_action]' -q -ra
```

## 29. `tests/test_remote.py::test_misbehaving_agents[returns_str-act-error]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[returns_str-act-error]' -q -ra
```

## 30. `tests/test_remote.py::test_misbehaving_agents[hard_crash-act-crash]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[hard_crash-act-crash]' -q -ra
```

## 31. `tests/test_remote.py::test_misbehaving_agents[gc_snoop-None-None]`

Previous result: **error**. Source line: 118.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_misbehaving_agents[gc_snoop-None-None]' -q -ra
```

## 32. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[conflicting_paths]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[conflicting_paths]' -q -ra
```

## 33. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[deep_json]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[deep_json]' -q -ra
```

## 34. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[huge_int]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[huge_int]' -q -ra
```

## 35. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[int64_wrap_shape]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[int64_wrap_shape]' -q -ra
```

## 36. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[negative_size_bomb]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[negative_size_bomb]' -q -ra
```

## 37. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[not_an_object]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[not_an_object]' -q -ra
```

## 38. `tests/test_remote.py::test_hostile_replies_become_protocol_failures[wrong_seq]`

Previous result: **failure**. Source line: 186.

Cause: The upstream test imports the Unix resource module, absent in native Windows.

Action: Run unchanged in Linux; preserve protocol failure and memory-growth checks.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_remote.py::test_hostile_replies_become_protocol_failures[wrong_seq]' -q -ra
```

## 39. `tests/test_validate.py::test_template_baseline_passes`

Previous result: **failure**. Source line: 21.

Cause: DoomAgent cannot initialize because genuine trained weights do not exist.

Action: Complete real PPO updates, validate and promote a checkpoint, then rerun this case.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_validate.py::test_template_baseline_passes' -q -ra
```

## 40. `tests/test_validate.py::test_crashing_agent_fails`

Previous result: **failure**. Source line: 33.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_validate.py::test_crashing_agent_fails' -q -ra
```

## 41. `tests/test_validate.py::test_optional_and_varargs_signatures_pass`

Previous result: **failure**. Source line: 43.

Cause: Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached.

Action: Pass Linux engine smoke, then rerun this exact case without changing its assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_validate.py::test_optional_and_varargs_signatures_pass' -q -ra
```

## 42. `tests/test_validate.py::test_cpu_and_ram_limits_are_emulated`

Previous result: **failure**. Source line: 73.

Cause: The upstream test calls os.sched_getaffinity, which native Windows does not provide.

Action: Run unchanged in Linux; verify real one-CPU affinity and peak RSS assertions.

Linux rerun: **NOT EXECUTED ON LINUX**.

```bash
python -m pytest 'tests/test_validate.py::test_cpu_and_ram_limits_are_emulated' -q -ra
```
