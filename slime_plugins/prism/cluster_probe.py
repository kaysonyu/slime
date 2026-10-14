"""Validate allocated GPU collectives and TensorBoard storage before model startup."""

import json
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import ray


@ray.remote(num_gpus=1, num_cpus=1)
class Probe:
    def rendezvous(self):
        from ray._private.services import get_node_ip_address

        with socket.socket() as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        return get_node_ip_address(), port

    def run(self, rank, world, address):
        import psutil

        import slime

        actual_source = Path(slime.__file__).resolve()
        if not actual_source.is_relative_to(Path(os.environ["PRISM_SLIME_SOURCE"]).resolve()):
            raise RuntimeError(f"Ray imported the wrong Slime source: {actual_source}")
        from argparse import Namespace

        import torch
        import torch.distributed as dist

        from slime.observability.tensorboard_utils import _TensorboardAdapter

        destination = Path(os.environ["TENSORBOARD_DIR"])
        assert destination.is_absolute() and str(destination).startswith("/inspire/qb-ilm2/")
        writer = _TensorboardAdapter(
            Namespace(
                use_tensorboard=True,
                tb_project_name="storage-probe",
                tb_experiment_name=str(rank),
                tensorboard_dir=str(destination),
            )
        )
        assert Path(writer._writer.log_dir) == destination
        writer.log({"validation/storage_probe": 1}, step=-1)
        writer._writer.flush()
        writer.finish()
        torch.cuda.set_device(0)
        started = time.perf_counter()
        dist.init_process_group("nccl", init_method=address, rank=rank, world_size=world, timeout=timedelta(minutes=3))
        results = []
        for count in (1024, 64 * 1024**2):
            value = torch.ones(count, device="cuda", dtype=torch.float32)
            for _ in range(2):
                value.fill_(1)
                dist.all_reduce(value)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(5):
                value.fill_(1)
                dist.all_reduce(value)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) / 5
            assert bool((value == world).all())
            results.append({"bytes": value.numel() * value.element_size(), "all_reduce_seconds": elapsed})
        interfaces = {
            name: [addr.address for addr in addrs if addr.family == socket.AF_INET]
            for name, addrs in psutil.net_if_addrs().items()
        }
        report = {
            "rank": rank,
            "host": socket.gethostname(),
            "gpu": str(torch.cuda.get_device_properties(0).uuid),
            "seconds": time.perf_counter() - started,
            "interfaces": interfaces,
            "all_reduce": results,
            "slime_module": str(actual_source),
            "tensorboard_dir": str(destination),
            "tensorboard_write_passed": True,
        }
        dist.destroy_process_group()
        return report


def main():
    ray.init(address="auto", logging_level="ERROR")
    world = int(sys.argv[1])
    actors = [Probe.remote() for _ in range(world)]
    try:
        host, port = ray.get(actors[0].rendezvous.remote())
        results = ray.get([actor.run.remote(rank, world, f"tcp://{host}:{port}") for rank, actor in enumerate(actors)])
        Path(sys.argv[2]).write_text(
            json.dumps({"passed": True, "world_size": world, "ranks": results}, indent=2) + "\n"
        )
        print(json.dumps({"nccl_probe_passed": True, "ranks": world}), flush=True)
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()
