"""Teacher queue lifetime and retry behavior without model or serving dependencies."""

import asyncio

import pytest

from slime.rollout.teacher_client import BatchedTeacherClient, TeacherClientOptions

NUM_GPUS = 0


class Client(BatchedTeacherClient):
    def __init__(self, send, **options):
        super().__init__(TeacherClientOptions(**options))
        self.send_request = send

    async def send_batch(self, route, requests):
        return await self.send_request(route, requests)


def test_close_cancels_pending_timers_and_rejects_new_requests():
    async def run():
        async def unexpected(*args):
            pytest.fail("pending requests were sent after close")

        client = Client(unexpected, batch_wait_ms=1000)
        pending = asyncio.create_task(client.score("route", 1))
        await asyncio.sleep(0)
        await client.close()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not client.timers and not client.pending and not client.tasks
        with pytest.raises(RuntimeError, match="closed"):
            await client.score("route", 2)

    asyncio.run(run())


@pytest.mark.parametrize("timeout_error", [TimeoutError, asyncio.TimeoutError])
def test_retry_and_receipt_are_owned_by_the_common_queue(timeout_error):
    async def run():
        attempts = []

        async def send(route, requests):
            attempts.append((route, requests))
            if len(attempts) == 1:
                raise timeout_error("transient")
            return [value * 2 for value in requests]

        client = Client(send, batch_size=2, batch_wait_ms=1000)
        first, second = await asyncio.gather(client.score("route", 3), client.score("route", 5))
        assert first.value == 6 and second.value == 10
        assert first.batch_size == 2 and first.attempts == 2
        assert len(attempts) == 2
        await client.close()

    asyncio.run(run())


def test_close_cancels_inflight_work_without_leaving_background_tasks():
    async def run():
        entered, exited = asyncio.Event(), asyncio.Event()

        async def send(route, requests):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()

        client = Client(send, batch_size=1)
        pending = asyncio.create_task(client.score("route", 1))
        await entered.wait()
        await client.close()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert exited.is_set() and not client.tasks

    asyncio.run(run())


@pytest.mark.parametrize("recover", [False, True])
def test_request_deadline_cancels_attempt_before_retry_and_reports_exhaustion(recover):
    async def run():
        events = []
        attempts = 0

        async def send(route, requests):
            nonlocal attempts
            attempts += 1
            attempt = attempts
            events.append(("start", attempt))
            try:
                if recover and attempt == 2:
                    return requests
                await asyncio.Event().wait()
            finally:
                events.append(("finish", attempt))

        client = Client(send, batch_size=1, request_timeout_s=0.05, max_attempts=2)
        try:
            if recover:
                receipt = await client.score("route", 7)
                assert receipt.value == 7 and receipt.attempts == 2
            else:
                with pytest.raises(asyncio.TimeoutError):
                    await client.score("route", 7)
            assert events == [("start", 1), ("finish", 1), ("start", 2), ("finish", 2)]
        finally:
            await client.close()
        assert not client.tasks and not client.pending

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
