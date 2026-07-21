"""The AiiDA process that executes a work graph."""

from __future__ import annotations

import functools
import logging
import typing as t

import kiwipy
from plumpy import process_comms
from plumpy.process_states import Continue, Wait
from plumpy.workchains import Stepper, _PropagateReturn

from aiida.common.lang import override
from aiida.engine.processes.exit_code import ExitCode
from aiida.engine.processes.process import ProcessState
from aiida.engine.processes.workchains.awaitable import Awaitable, AwaitableTarget
from aiida.engine.processes.workchains.workchain import WorkChain, WorkChainSpec

from aiida_workgraph.engine.error_handler_manager import ErrorHandlerManager
from aiida_workgraph.engine.stepper import DagStepper
from aiida_workgraph.engine.task_manager import TaskManager
from aiida_workgraph.enums import TaskActionMessage
from aiida_workgraph.orm.workgraph import WorkGraphNode

if t.TYPE_CHECKING:
    from aiida.engine.runners import Runner
    from aiida_workgraph import WorkGraph

__all__ = ('WorkGraphProcess', 'WorkGraphSpec')


class WorkGraphSpec(WorkChainSpec):
    WORKGRAPH_DATA_KEY = 'workgraph_data'


class WorkGraphProcess(WorkChain):
    """Execute a work graph, scheduling its tasks by their data dependencies.

    A work chain declares its execution order up front as an outline; a work graph derives it from the links
    between tasks, which are only known once the graph is built. Everything else a work chain provides (context,
    awaitables, checkpointing, node lifecycle) applies unchanged, so this supplies a :class:`DagStepper` through the
    stepper hooks and inherits the rest.

    The three places it still departs from :class:`~aiida.engine.processes.workchains.workchain.WorkChain` all trace
    back to a single difference: a work chain waits for everything a step launched before starting the next one,
    whereas here independent branches must stay in flight together. See :meth:`_do_step`,
    :meth:`_action_awaitables` and :meth:`_on_awaitable_finished`.
    """

    # Narrowing the node and spec classes is how every AiiDA process specialises its base; mypy sees plain mutable
    # class attributes and flags the covariance.
    _node_class = WorkGraphNode  # type: ignore[mutable-override]
    _spec_class = WorkGraphSpec  # type: ignore[mutable-override]

    def __init__(
        self,
        inputs: dict[str, t.Any] | None = None,
        logger: logging.Logger | None = None,
        runner: 'Runner' | None = None,
        enable_persistence: bool = True,
    ) -> None:
        super().__init__(inputs, logger, runner, enable_persistence=enable_persistence)
        self._init_runtime_state()
        self._init_managers()

    def _init_runtime_state(self) -> None:
        """Initialise the state that is rebuilt on every load rather than restored from the checkpoint."""
        self._wg: 'WorkGraph' | None = None
        # Awaitables whose completion callback is already registered with the runner. Callbacks do not survive a
        # checkpoint, so this must start empty on every load, which is why it is not kept in the context.
        self._registered_awaitable_pks: set[int] = set()

    def _init_managers(self) -> None:
        self.task_manager = TaskManager(self.logger, self.runner, self)
        self.error_handler_manager = ErrorHandlerManager(self, self.logger)

    @classmethod
    def define(cls, spec: WorkGraphSpec) -> None:  # type: ignore[override]
        super().define(spec)
        spec.input_namespace(
            'graph_inputs',
            dynamic=True,
            required=False,
            help='Graph level inputs',
        )
        spec.input_namespace(
            'tasks',
            dynamic=True,
            required=False,
            help='Tasks inputs',
        )
        spec.input_namespace(
            spec.WORKGRAPH_DATA_KEY,
            dynamic=True,
            required=False,
            help='WorkGraph data',
        )
        spec.exit_code(2, 'ERROR_SUBPROCESS', message='A subprocess has failed.')
        spec.outputs.dynamic = True
        #
        spec.exit_code(201, 'UNKNOWN_MESSAGE_TYPE', message='The message type is unknown.')
        spec.exit_code(202, 'UNKNOWN_TASK_TYPE', message='The task type is unknown.')
        #
        spec.exit_code(
            301,
            'OUTPUS_NOT_MATCH_RESULTS',
            message='The outputs of the process do not match the results.',
        )
        spec.exit_code(
            302,
            'TASK_FAILED',
            message='Some of the tasks failed.',
        )
        spec.exit_code(
            303,
            'TASK_NON_ZERO_EXIT_STATUS',
            message='Some of the tasks exited with non-zero status.',
        )

    @property
    def wg(self) -> 'WorkGraph':
        """The work graph being executed, rebuilt from the context the first time it is needed after a reload."""
        if self._wg is None:
            from aiida_workgraph import WorkGraph

            self._wg = WorkGraph.from_dict(self.ctx._wgdata)
        return self._wg

    def set_workgraph_data(self, wgdata: dict[str, t.Any]) -> None:
        """Install the graph data, keeping the live graph and the checkpointed copy in step."""
        from aiida_workgraph import WorkGraph

        self.ctx._wgdata = wgdata
        self._wg = WorkGraph.from_dict(wgdata)

    def _create_stepper(self) -> Stepper:
        stepper = DagStepper(self)  # type: ignore[arg-type]
        stepper.setup()
        return stepper

    def _recreate_stepper(self, saved_state: t.Any) -> Stepper:
        """Restore the stepper after a checkpoint.

        ``saved_state`` is unused: :class:`DagStepper` holds no state of its own, everything it needs lives in the
        context, which the base class has already restored. Notably :meth:`DagStepper.setup` must not run again,
        as it would reset the execution bookkeeping and rerun the finished tasks.
        """
        return DagStepper(self)  # type: ignore[arg-type]

    @override
    def load_instance_state(self, saved_state: t.MutableMapping[str, t.Any], load_context: t.Any) -> None:
        from aiida.orm.utils.log import create_logger_adapter

        # `WorkChain.load_instance_state` re-registers the awaitable callbacks before returning, so the runtime
        # state it consults has to be in place first.
        self._init_runtime_state()

        # `no-untyped-call` here and below: both are unannotated aiida-core internals.
        super().load_instance_state(saved_state, load_context)  # type: ignore[no-untyped-call]

        # TODO: avoid hardcoding the logger
        self.node._logger = logging.getLogger('aiida.orm.nodes.process.workflow.workchain.WorkChainNode')  # type: ignore[assignment]
        # First time the property is called after the node is stored, create the logger adapter
        self.node._logger_adapter = create_logger_adapter(self.node._logger, self.node)  # type: ignore[no-untyped-call]
        self.set_logger(self.node._logger_adapter)

        self._init_managers()

    def _do_step(self) -> t.Any:
        """Advance the graph by one step.

        Deliberately not delegating to :meth:`WorkChain._do_step`, which opens by clearing ``self._awaitables``.
        That clearing is what makes an outline step wait for everything it launched; a work graph must keep its
        awaitables so that tasks launched in earlier steps stay in flight while later ones start.
        """
        result: t.Any = None

        try:
            assert self._stepper is not None
            finished, result = self._stepper.step()
        except _PropagateReturn as exception:
            finished, result = True, exception.exit_code

        if finished or isinstance(result, ExitCode):
            return result

        if self._awaitables:
            return Wait(self._do_step, 'Waiting before next step')

        return Continue(self._do_step)

    def _action_awaitables(self) -> None:
        """Register a completion callback for each awaitable that does not already have one.

        :class:`~aiida.engine.processes.workchains.workchain.WorkChain` empties its awaitables every step, so by the
        time it gets here they are always new and it can register unconditionally. A work graph carries its
        awaitables across steps, so the same one is seen again on each pass through the waiting state and would
        otherwise collect a further callback every time.
        """
        for awaitable in self._awaitables:
            if awaitable.pk in self._registered_awaitable_pks:
                continue
            if awaitable.target != AwaitableTarget.PROCESS:
                raise AssertionError(f"invalid awaitable target '{awaitable.target}'")
            callback = functools.partial(self.call_soon, self._on_awaitable_finished, awaitable)
            self.runner.call_on_process_finish(awaitable.pk, callback)
            self._registered_awaitable_pks.add(awaitable.pk)

        # `WorkChain` records "Waiting for child processes: ..." only as the process status. Surface it in the
        # report log as well, so it shows up in `verdi process report` when a graph pauses for its children.
        if self._awaitables:
            self.report(f'Process status: {self.status}')

    def _on_awaitable_finished(self, awaitable: Awaitable) -> None:
        """Resolve a finished awaitable, record the outcome on its task, and resume.

        Unlike :class:`~aiida.engine.processes.workchains.workchain.WorkChain`, this resumes while other awaitables
        are still outstanding: a finished task can unblock its dependents no matter what else is running. Resuming
        is conditional on still being in the waiting state, because several awaitables finishing in the same batch
        would otherwise each try to resume an already-running process.

        :param awaitable: the awaitable whose target process has terminated
        """
        from aiida.common import exceptions
        from aiida.orm.utils import load_node

        self.logger.info('received callback that awaitable %d has terminated', awaitable.pk)

        try:
            node = load_node(awaitable.pk)
        except (exceptions.MultipleObjectsError, exceptions.NotExistent):
            msg = f'provided pk<{awaitable.pk}> could not be resolved to a valid Node instance'
            raise ValueError(msg)

        if awaitable.outputs:
            value: t.Any = {entry.link_label: entry.node for entry in node.base.links.get_outgoing()}
        else:
            value = node

        self._resolve_awaitable(awaitable, value)
        self._registered_awaitable_pks.discard(awaitable.pk)
        self.task_manager.state_manager.update_task_state(awaitable.key)

        if self.state == ProcessState.WAITING:
            self.resume()

    def _build_process_label(self) -> str:
        """Use the workgraph name as the process label."""
        return f'WorkGraph<{self.inputs[WorkGraphSpec.WORKGRAPH_DATA_KEY]["name"]}>'

    def on_create(self) -> None:
        """Called when a Process is created."""
        from aiida_workgraph.utils import save_workgraph_data

        super().on_create()
        raw_inputs = dict(self.inputs)
        self.node.label = raw_inputs[WorkGraphSpec.WORKGRAPH_DATA_KEY]['name']
        save_workgraph_data(self.node, raw_inputs)

    def apply_action(self, msg: TaskActionMessage) -> None:
        if msg['catalog'] == 'task':
            self.task_manager.action_manager.apply_task_actions(msg)
        else:
            self.report(f'Unknow message type {msg}')

    def message_receive(self, _comm: kiwipy.Communicator, msg: t.Dict[str, t.Any]) -> t.Any:
        """Handle the work-graph-specific ``custom`` intent and defer every standard intent to the base class.

        Only the ``custom`` intent, which carries task actions such as pausing or skipping an individual task, is
        particular to a work graph. The rest belong to plumpy's message protocol, which decides such things as
        which key holds the pause text and whether a kill is forced; that protocol changes, so reimplementing it
        here means silently drifting out of step with it.

        :param _comm: the communicator that sent the message
        :param msg: the message
        :return: the outcome of processing the message, sent back as the response to the sender
        """
        if msg[process_comms.INTENT_KEY] == 'custom':
            return self._schedule_rpc(self.apply_action, msg=msg)

        return super().message_receive(_comm, msg)
