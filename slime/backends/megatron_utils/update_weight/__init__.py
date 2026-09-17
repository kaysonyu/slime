"""Complete-weight NCCL publication to separately managed Omni stages."""


def create_weight_updater(args, model):
    from .update_weight_from_distributed import UpdateWeightFromDistributed

    return UpdateWeightFromDistributed(args, model)
