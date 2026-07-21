from aiida_workgraph import WorkGraph, task
import asyncio
from aiida.cmdline.utils.common import get_workchain_report


def test_async_decorator(decorated_add):
    @task()
    async def async_func(x, y):
        n = 2
        while n > 0:
            n -= 1
            await asyncio.sleep(0.5)
        return x + y

    wg = WorkGraph(name='test_async_functioin')
    async_func1 = wg.add_task(async_func, 'async_func1', x=1, y=2)
    add1 = wg.add_task(decorated_add, 'add1', x=1, y=async_func1.outputs.result)
    wg.run()
    assert add1.outputs.result.value == 4
    report = get_workchain_report(wg.process, 'REPORT')
    assert 'Waiting for child processes: ' in report


def test_ready_task_starts_before_slow_sibling_finishes():
    """A task whose inputs are ready starts while an unrelated, slower task is still running.

    This is what separates scheduling on data dependencies from stepping an outline. An outline step waits for
    everything it launched before the next step begins, which would hold ``child`` back until ``slow`` had also
    finished. Here ``child`` depends only on ``fast``, so it must start as soon as ``fast`` resolves, with ``slow``
    still in flight. Two concurrent async tasks in the runner's own event loop make the ordering deterministic:
    ``fast`` returns immediately, ``slow`` only after a delay long enough that ``child`` is guaranteed to have
    started first.
    """

    @task()
    async def sleep_return(x, seconds):
        await asyncio.sleep(seconds)
        return x

    @task()
    async def passthrough(x):
        return x

    wg = WorkGraph(name='test_ready_task_starts_before_slow_sibling_finishes')
    wg.add_task(sleep_return, 'slow', x=1, seconds=5)
    fast = wg.add_task(sleep_return, 'fast', x=2, seconds=0)
    wg.add_task(passthrough, 'child', x=fast.outputs.result)
    wg.run()

    report = get_workchain_report(wg.process, 'REPORT')
    # ``child`` becomes ready (after ``fast`` resolves) before ``slow`` reports it has finished.
    assert report.index('tasks ready to run: child') < report.index('Task: slow, type: PYFUNCTION, finished')
