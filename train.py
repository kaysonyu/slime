import logging

import ray

from slime.data.checkpoint import save_checkpoint
from slime.observability.config_snapshot import write_resolved_configuration
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.observability.phases import PhaseRecorder
from slime.ray.placement_group import create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.cleanup import Cleanup
from slime.utils.misc import should_run_periodic_action


def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    with PhaseRecorder(getattr(args, "phase_trace_path", None)) as phases:
        release_train = args.release_train
        cleanup_interval = getattr(args, "resident_cleanup_interval", 1)
        if cleanup_interval < 1:
            raise ValueError("resident_cleanup_interval must be positive")

        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())

        # Always push actor weights to rollout once weights are loaded.
        with phases.measure("update_weights", args.start_rollout_id - 1):
            actor_model.update_weights()
        # Reattached async producers stay paused until restored weights are installed.
        # This startup notification resumes them once, before entering the train loop.
        ray.get(rollout_manager.training_ready.remote())

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # special case for eval-only
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))

        def offload_train(actor_trains_this_step):
            # Training or its following checkpoint offloads the model when configured.
            if not args.offload_train and (rollout_id + 1) % cleanup_interval == 0:
                if not args.use_critic or actor_trains_this_step:
                    actor_model.clear_memory()
                else:
                    critic_model.clear_memory()

        # train loop.
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                with phases.measure("eval", rollout_id):
                    ray.get(rollout_manager.eval.remote(rollout_id))

            with phases.measure("generate", rollout_id):
                rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

            if args.offload_rollout:
                with phases.measure("offload", rollout_id):
                    ray.get(rollout_manager.offload.remote())

            if release_train:
                actor_model.create()

            save_this_rollout = release_train or should_run_periodic_action(
                rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
            )
            actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
            with phases.measure("async_train", rollout_id):
                if args.use_critic:
                    value_refs = critic_model.async_train(
                        rollout_id, rollout_data_ref, offload_after_train=actor_trains or not save_this_rollout
                    )
                    if actor_trains:
                        ray.get(
                            actor_model.async_train(
                                rollout_id,
                                rollout_data_ref,
                                external_data=value_refs,
                                offload_after_train=not save_this_rollout,
                            )
                        )
                    else:
                        ray.get(value_refs)
                else:
                    ray.get(
                        actor_model.async_train(
                            rollout_id, rollout_data_ref, offload_after_train=not save_this_rollout
                        )
                    )

            # Runtime completion releases queue capacity, without claiming that the
            # model/optimizer or this data progress have a durable joint checkpoint.
            with phases.measure("training_completed", rollout_id):
                ray.get(rollout_manager.training_completed.remote(rollout_id))

            if save_this_rollout:
                with phases.measure("checkpoint", rollout_id):
                    save_checkpoint(
                        args,
                        rollout_id,
                        actor_model,
                        critic_model,
                        rollout_manager,
                        actor_trains=actor_trains,
                        restore_plan=restore_plan,
                    )

            with phases.measure("clear_memory", rollout_id):
                offload_train(actor_trains)
            if args.offload_rollout and not release_train:
                with phases.measure("onload_weights", rollout_id):
                    ray.get(rollout_manager.onload_weights.remote())
            with phases.measure("pause_rollout_admission", rollout_id):
                was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
            with phases.measure("update_weights", rollout_id):
                actor_model.update_weights()
            # Final evaluation uses the synchronized engines directly. Keep training
            # producers paused when no later rollout will consume their new work.
            if rollout_id + 1 < args.num_rollout:
                with phases.measure("resume_rollout_admission", rollout_id):
                    ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))

            if args.offload_rollout:
                with phases.measure("onload_kv", rollout_id):
                    ray.get(rollout_manager.onload_kv.remote())

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                with phases.measure("eval", rollout_id):
                    ray.get(rollout_manager.eval.remote(rollout_id))


def main(args, restore_plan):
    """Own the attempt's exit policy; actors only clean up their own resources."""
    with Cleanup() as cleanup:
        init_tracking(args)
        startup = None
        try:
            # Create or reattach rollout resources, then run this training attempt.
            startup = create_rollout_manager(args, restore_plan=restore_plan)
            write_resolved_configuration(startup.args)
            train(
                startup.args, startup.placements, startup.manager, startup.num_rollout_per_epoch, startup.restore_plan
            )
        except BaseException:
            # Failures and interruptions retain serving and replay data for a manual
            # restart. Detach this driver's trainers instead of disposing the session.
            if startup is not None:
                try:
                    startup.close(failed=True)
                except Exception:
                    logging.getLogger(__name__).exception("Failed to clean up training attempt")
            raise
        else:
            startup.close(failed=False)
        finally:
            # Finish this driver's tracking on success, startup failure, or interruption.
            cleanup.run("finish driver tracking", finish_tracking, args)


if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    main(args, restore_plan)
