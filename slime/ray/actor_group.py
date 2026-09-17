"""Dedicated Megatron ranks placed with slime's existing Ray GPU ordering."""

import os

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, add_default_ray_env_vars


class RayTrainGroup:
    def __init__(self, args, num_nodes, num_gpus_per_node, pg):
        self.args = args
        self.world_size = num_nodes * num_gpus_per_node
        self.pg = pg
        self._actor_handlers = []

    def create(self, rollout_manager):
        from slime.backends.megatron_utils.actor import MegatronTrainRayActor

        group, ordered_bundles, _ = self.pg
        env = {
            # NCCL allocation settings must match the separately launched Omni service.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }
        actor_type = ray.remote(MegatronTrainRayActor)
        address = port = None
        for rank in range(self.world_size):
            actor = actor_type.options(
                num_cpus=1,
                num_gpus=1,
                runtime_env={"env_vars": add_default_ray_env_vars(env)},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_bundle_index=ordered_bundles[rank],
                ),
            ).remote(self.world_size, rank, address, port)
            self._actor_handlers.append(actor)
            if rank == 0:
                address, port = ray.get(actor.get_master_addr_and_port.remote())
        starts = ray.get([actor.init.remote(self.args) for actor in self._actor_handlers])
        ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
        return starts

    def async_train(self, rollout_id, rollout_data_ref):
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    def update_weights(self):
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def dispose(self):
        if self._actor_handlers:
            ray.get([actor.dispose.remote() for actor in self._actor_handlers], timeout=60)

    def release(self):
        actors, self._actor_handlers = self._actor_handlers, []
        for actor in actors:
            ray.kill(actor, no_restart=True)
