"""Synchronous Omni rollout, Megatron update, checkpoint and weight publication."""

import logging
import signal

import ray
from ray.util.placement_group import remove_placement_group

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    owns_ray = not ray.is_initialized()
    rollout_manager = actor = None
    pgs = {}
    if owns_ray:
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init(
                address="local",
                num_gpus=0 if args.debug_rollout_only else args.actor_num_gpus_per_node,
                include_dashboard=False,
            )
    try:
        pgs = create_placement_groups(args)
        init_tracking(args)
        rollout_manager, per_epoch = create_rollout_manager(args)
        if not args.debug_rollout_only:
            actor = create_training_models(args, pgs, rollout_manager)
            actor.update_weights()
        else:
            args.start_rollout_id = args.start_rollout_id or 0
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(0))
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            batch = ray.get(rollout_manager.generate.remote(rollout_id))
            if actor is not None:
                ray.get(actor.async_train(rollout_id, batch))
            if args.save and should_run_periodic_action(rollout_id, args.save_interval, per_epoch, args.num_rollout):
                if actor is not None:
                    actor.save_model(rollout_id, force_sync=True)
                ray.get(rollout_manager.save.remote(rollout_id))
            if actor is not None:
                actor.update_weights()
            if should_run_periodic_action(rollout_id, args.eval_interval, per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))
    finally:
        if actor is not None:
            try:
                actor.dispose()
            except Exception:
                logging.getLogger(__name__).exception("Training-group cleanup failed")
            finally:
                actor.release()
        if rollout_manager is not None:
            try:
                ray.get(rollout_manager.dispose.remote(), timeout=60)
            except Exception:
                logging.getLogger(__name__).exception("Rollout cleanup failed")
            finally:
                ray.kill(rollout_manager)
        if pgs and pgs["actor"][0] is not None:
            remove_placement_group(pgs["actor"][0])
        finish_tracking(args)
        if owns_ray:
            ray.shutdown()


if __name__ == "__main__":

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    train(parse_args())
