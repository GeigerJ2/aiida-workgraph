"""AiiDA workflow components: WorkGraph."""
from __future__ import annotations

import asyncio
import collections.abc
import functools
import logging
import typing as t

from plumpy import process_comms
from plumpy.persistence import auto_persist
from plumpy.process_states import Continue, Wait
from plumpy.workchains import _PropagateReturn
import kiwipy

from aiida.common import exceptions
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import override
from aiida import orm
from aiida.orm import load_node, Node, ProcessNode, WorkChainNode
from aiida.orm.utils.serialize import deserialize_unsafe, serialize

from aiida.engine.processes.exit_code import ExitCode
from aiida.engine.processes.process import Process

from aiida.engine.processes.workchains.awaitable import (
    Awaitable,
    AwaitableAction,
    AwaitableTarget,
    construct_awaitable,
)
from aiida.engine.processes.workchains.workchain import Protect, WorkChainSpec
from aiida.engine import run_get_node
from aiida_workgraph.utils import create_and_pause_process
from aiida_workgraph.task import Task
from aiida_workgraph.utils import get_nested_dict, update_nested_dict
from aiida_workgraph.executors.monitors import monitor

if t.TYPE_CHECKING:
    from aiida.engine.runners import Runner  # pylint: disable=unused-import

__all__ = "WorkGraph"


MAX_NUMBER_AWAITABLES_MSG = "The maximum number of subprocesses has been reached: {}. Cannot launch the job: {}."


@auto_persist("_awaitables")
class WorkGraphEngine(Process, metaclass=Protect):
    """The `WorkGraph` class is used to construct workflows in AiiDA."""

    # used to create a process node that represents what happened in this process.
    _node_class = WorkChainNode
    _spec_class = WorkChainSpec
    _CONTEXT = "CONTEXT"

    def __init__(
        self,
        inputs: dict | None = None,
        logger: logging.Logger | None = None,
        runner: "Runner" | None = None,
        enable_persistence: bool = True,
    ) -> None:
        """Construct a WorkGraph instance.

        :param inputs: work graph inputs
        :param logger: aiida logger
        :param runner: work graph runner
        :param enable_persistence: whether to persist this work graph

        """

        super().__init__(inputs, logger, runner, enable_persistence=enable_persistence)

        self._awaitables: list[Awaitable] = []
        self._context = AttributeDict()

    @classmethod
    def define(cls, spec: WorkChainSpec) -> None:
        super().define(spec)
        spec.input("input_file", valid_type=orm.SinglefileData, required=False)
        spec.input_namespace(
            "wg", dynamic=True, required=False, help="WorkGraph inputs"
        )
        spec.input_namespace("input_tasks", dynamic=True, required=False)
        spec.exit_code(2, "ERROR_SUBPROCESS", message="A subprocess has failed.")

        spec.outputs.dynamic = True

        spec.output(
            "execution_count",
            valid_type=orm.Int,
            required=False,
            help="The number of time the WorkGraph runs.",
        )
        #
        spec.exit_code(
            201, "UNKNOWN_MESSAGE_TYPE", message="The message type is unknown."
        )
        spec.exit_code(202, "UNKNOWN_TASK_TYPE", message="The task type is unknown.")
        #
        spec.exit_code(
            301,
            "OUTPUS_NOT_MATCH_RESULTS",
            message="The outputs of the process do not match the results.",
        )
        spec.exit_code(
            302,
            "TASK_FAILED",
            message="Some of the tasks failed.",
        )
        spec.exit_code(
            303,
            "TASK_NON_ZERO_EXIT_STATUS",
            message="Some of the tasks exited with non-zero status.",
        )

    @property
    def ctx(self) -> AttributeDict:
        """Get the context."""
        return self._context

    @override
    def save_instance_state(
        self, out_state: t.Dict[str, t.Any], save_context: t.Any
    ) -> None:
        """Save instance state.

        :param out_state: state to save in

        :param save_context:
        :type save_context: :class:`!plumpy.persistence.LoadSaveContext`

        """
        super().save_instance_state(out_state, save_context)
        # Save the context
        out_state[self._CONTEXT] = self.ctx

    @override
    def load_instance_state(
        self, saved_state: t.Dict[str, t.Any], load_context: t.Any
    ) -> None:
        super().load_instance_state(saved_state, load_context)
        # Load the context
        self._context = saved_state[self._CONTEXT]
        self._temp = {"awaitables": {}}

        self.set_logger(self.node.logger)

        if self._awaitables:
            # For the "ascyncio.tasks.Task" awaitable, because there are only in-memory,
            # we need to reset the tasks and so that they can be re-run again.
            should_resume = False
            for awaitable in self._awaitables:
                if awaitable.target == "asyncio.tasks.Task":
                    self._resolve_awaitable(awaitable, None)
                    self.report(f"reset awaitable task: {awaitable.key}")
                    self.reset_task(awaitable.key)
                    should_resume = True
            if should_resume:
                self._update_process_status()
                self.resume()
            # For other awaitables, because they exist in the db, we only need to re-register the callbacks
            self.ctx._awaitable_actions = []
            self._action_awaitables()

    def _resolve_nested_context(self, key: str) -> tuple[AttributeDict, str]:
        """
        Returns a reference to a sub-dictionary of the context and the last key,
        after resolving a potentially segmented key where required sub-dictionaries are created as needed.

        :param key: A key into the context, where words before a dot are interpreted as a key for a sub-dictionary
        """
        ctx = self.ctx
        ctx_path = key.split(".")

        for index, path in enumerate(ctx_path[:-1]):
            try:
                ctx = ctx[path]
            except KeyError:  # see below why this is the only exception we have to catch here
                ctx[
                    path
                ] = AttributeDict()  # create the sub-dict and update the context
                ctx = ctx[path]
                continue

            # Notes:
            # * the first ctx (self.ctx) is guaranteed to be an AttributeDict, hence the post-"dereference" checking
            # * the values can be many different things: on insertion they are either AtrributeDict, List or Awaitables
            #   (subclasses of AttributeDict) but after resolution of an Awaitable this will be the value itself
            # * assumption: a resolved value is never a plain AttributeDict, on the other hand if a resolved Awaitable
            #   would be an AttributeDict we can append things to it since the order of tasks is maintained.
            if type(ctx) != AttributeDict:  # pylint: disable=C0123
                raise ValueError(
                    f"Can not update the context for key `{key}`: "
                    f' found instance of `{type(ctx)}` at `{".".join(ctx_path[:index + 1])}`, expected AttributeDict'
                )

        return ctx, ctx_path[-1]

    def _insert_awaitable(self, awaitable: Awaitable) -> None:
        """Insert an awaitable that should be terminated before before continuing to the next step.

        :param awaitable: the thing to await
        """
        ctx, key = self._resolve_nested_context(awaitable.key)

        # Already assign the awaitable itself to the location in the context container where it is supposed to end up
        # once it is resolved. This is especially important for the `APPEND` action, since it needs to maintain the
        # order, but the awaitables will not necessarily be resolved in the order in which they are added. By using the
        # awaitable as a placeholder, in the `_resolve_awaitable`, it can be found and replaced by the resolved value.
        if awaitable.action == AwaitableAction.ASSIGN:
            ctx[key] = awaitable
        elif awaitable.action == AwaitableAction.APPEND:
            ctx.setdefault(key, []).append(awaitable)
        else:
            raise AssertionError(f"Unsupported awaitable action: {awaitable.action}")

        self._awaitables.append(
            awaitable
        )  # add only if everything went ok, otherwise we end up in an inconsistent state
        self._update_process_status()

    def _resolve_awaitable(self, awaitable: Awaitable, value: t.Any) -> None:
        """Resolve an awaitable.

        Precondition: must be an awaitable that was previously inserted.

        :param awaitable: the awaitable to resolve
        :param value: the value to assign to the awaitable
        """
        ctx, key = self._resolve_nested_context(awaitable.key)

        if awaitable.action == AwaitableAction.ASSIGN:
            ctx[key] = value
        elif awaitable.action == AwaitableAction.APPEND:
            # Find the same awaitable inserted in the context
            container = ctx[key]
            for index, placeholder in enumerate(container):
                if (
                    isinstance(placeholder, Awaitable)
                    and placeholder.pk == awaitable.pk
                ):
                    container[index] = value
                    break
            else:
                raise AssertionError(
                    f"Awaitable `{awaitable.pk} was not in `ctx.{awaitable.key}`"
                )
        else:
            raise AssertionError(f"Unsupported awaitable action: {awaitable.action}")

        awaitable.resolved = True
        # remove awaitabble from the list
        self._awaitables = [a for a in self._awaitables if a.pk != awaitable.pk]

        if not self.has_terminated():
            # the process may be terminated, for example, if the process was killed or excepted
            # then we should not try to update it
            self._update_process_status()

    @Protect.final
    def to_context(self, **kwargs: Awaitable | ProcessNode) -> None:
        """Add a dictionary of awaitables to the context.

        This is a convenience method that provides syntactic sugar, for a user to add multiple intersteps that will
        assign a certain value to the corresponding key in the context of the work graph.
        """
        for key, value in kwargs.items():
            awaitable = construct_awaitable(value)
            awaitable.key = key
            self._insert_awaitable(awaitable)

    def _update_process_status(self) -> None:
        """Set the process status with a message accounting the current sub processes that we are waiting for."""
        if self._awaitables:
            status = f"Waiting for child processes: {', '.join([str(_.pk) for _ in self._awaitables])}"
            self.node.set_process_status(status)
        else:
            self.node.set_process_status(None)

    @override
    def run(self) -> t.Any:
        self.setup()
        return self._do_step()

    def _do_step(self) -> t.Any:
        """Execute the next step in the workgraph and return the result.

        If any awaitables were created, the process will enter in the Wait state,
        otherwise it will go to Continue.
        """
        # we will not remove the awaitables here,
        # we resume the workgraph in the callback function even
        # there are some awaitables left
        # self._awaitables = []
        result: t.Any = None

        try:
            self.continue_workgraph()
        except _PropagateReturn as exception:
            finished, result = True, exception.exit_code
        else:
            finished, result = self.is_workgraph_finished()

        # If the workgraph is finished or the result is an ExitCode, we exit by returning
        if finished:
            if isinstance(result, ExitCode):
                return result
            else:
                return self.finalize()

        if self._awaitables:
            return Wait(self._do_step, "Waiting before next step")

        return Continue(self._do_step)

    def _store_nodes(self, data: t.Any) -> None:
        """Recurse through a data structure and store any unstored nodes that are found along the way

        :param data: a data structure potentially containing unstored nodes
        """
        if isinstance(data, Node) and not data.is_stored:
            data.store()
        elif isinstance(data, collections.abc.Mapping):
            for _, value in data.items():
                self._store_nodes(value)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            for value in data:
                self._store_nodes(value)

    @override
    @Protect.final
    def on_exiting(self) -> None:
        """Ensure that any unstored nodes in the context are stored, before the state is exited

        After the state is exited the next state will be entered and if persistence is enabled, a checkpoint will
        be saved. If the context contains unstored nodes, the serialization necessary for checkpointing will fail.
        """
        super().on_exiting()
        try:
            self._store_nodes(self.ctx)
        except Exception:  # pylint: disable=broad-except
            # An uncaught exception here will have bizarre and disastrous consequences
            self.logger.exception("exception in _store_nodes called in on_exiting")

    @Protect.final
    def on_wait(self, awaitables: t.Sequence[t.Awaitable]):
        """Entering the WAITING state."""
        super().on_wait(awaitables)
        if self._awaitables:
            self._action_awaitables()
            self.report("Process status: {}".format(self.node.process_status))
        else:
            self.call_soon(self.resume)

    def _action_awaitables(self) -> None:
        """Handle the awaitables that are currently registered with the work chain.

        Depending on the class type of the awaitable's target a different callback
        function will be bound with the awaitable and the runner will be asked to
        call it when the target is completed
        """
        for awaitable in self._awaitables:
            # if the waitable already has a callback, skip
            if awaitable.pk in self.ctx._awaitable_actions:
                continue
            if awaitable.target == AwaitableTarget.PROCESS:
                callback = functools.partial(
                    self.call_soon, self._on_awaitable_finished, awaitable
                )
                self.runner.call_on_process_finish(awaitable.pk, callback)
                self.ctx._awaitable_actions.append(awaitable.pk)
            elif awaitable.target == "asyncio.tasks.Task":
                # this is a awaitable task, the callback function is already set
                self.ctx._awaitable_actions.append(awaitable.pk)
            else:
                assert f"invalid awaitable target '{awaitable.target}'"

    def _on_awaitable_finished(self, awaitable: Awaitable) -> None:
        """Callback function, for when an awaitable process instance is completed.

        The awaitable will be effectuated on the context of the work chain and removed from the internal list. If all
        awaitables have been dealt with, the work chain process is resumed.

        :param awaitable: an Awaitable instance
        """
        self.logger.debug(f"Awaitable {awaitable.key} finished.")

        if isinstance(awaitable.pk, int):
            self.logger.info(
                "received callback that awaitable with key {} and pk {} has terminated".format(
                    awaitable.key, awaitable.pk
                )
            )
            try:
                node = load_node(awaitable.pk)
            except (exceptions.MultipleObjectsError, exceptions.NotExistent):
                raise ValueError(
                    f"provided pk<{awaitable.pk}> could not be resolved to a valid Node instance"
                )

            if awaitable.outputs:
                value = {
                    entry.link_label: entry.node
                    for entry in node.base.links.get_outgoing()
                }
            else:
                value = node  # type: ignore
        else:
            # In this case, the pk and key are the same.
            self.logger.info(
                "received callback that awaitable {} has terminated".format(
                    awaitable.key
                )
            )
            try:
                # if awaitable is cancelled, the result is None
                if awaitable.cancelled():
                    self.set_task_state_info(awaitable.key, "state", "KILLED")
                    # set child tasks state to SKIPPED
                    self.set_tasks_state(
                        self.ctx._connectivity["child_node"][awaitable.key], "SKIPPED"
                    )
                    self.report(f"Task: {awaitable.key} cancelled.")
                else:
                    results = awaitable.result()
                    self.update_normal_task_state(awaitable.key, results)
            except Exception as e:
                self.logger.error(f"Error in awaitable {awaitable.key}: {e}")
                self.set_task_state_info(awaitable.key, "state", "FAILED")
                # set child tasks state to SKIPPED
                self.set_tasks_state(
                    self.ctx._connectivity["child_node"][awaitable.key], "SKIPPED"
                )
                self.report(f"Task: {awaitable.key} failed.")
                self.run_error_handlers(awaitable.key)
            value = None

        self._resolve_awaitable(awaitable, value)

        # node finished, update the task state and result
        # udpate the task state
        self.update_task_state(awaitable.key)
        # try to resume the workgraph, if the workgraph is already resumed
        # by other awaitable, this will not work
        try:
            self.resume()
        except Exception as e:
            print(e)

    def _build_process_label(self) -> str:
        """Use the workgraph name as the process label."""
        return f"WorkGraph<{self.inputs.wg['name']}>"

    def on_create(self) -> None:
        """Called when a Process is created."""
        from aiida_workgraph.utils.analysis import WorkGraphSaver

        super().on_create()
        wgdata = self.inputs.wg._dict
        restart_process = (
            orm.load_node(wgdata["restart_process"].value)
            if wgdata.get("restart_process")
            else None
        )
        saver = WorkGraphSaver(self.node, wgdata, restart_process=restart_process)
        saver.save()
        self.node.label = wgdata["name"]

    def setup(self) -> None:
        """Setup the variables in the context."""
        # track if the awaitable callback is added to the runner
        self.ctx._awaitable_actions = []
        self.ctx._new_data = {}
        self.ctx._executed_tasks = []
        # read the latest workgraph data
        wgdata = self.read_wgdata_from_base()
        self.init_ctx(wgdata)
        #
        self.ctx._msgs = []
        self.ctx._execution_count = 1
        # init task results
        self.set_task_results()
        # data not to be persisted, because they are not serializable
        self._temp = {"awaitables": {}}
        # while workgraph
        if self.ctx._workgraph["workgraph_type"].upper() == "WHILE":
            self.ctx._max_iteration = self.ctx._workgraph.get("max_iteration", 1000)
            should_run = self.check_while_conditions()
            if not should_run:
                self.set_tasks_state(self.ctx._tasks.keys(), "SKIPPED")
        # for workgraph
        if self.ctx._workgraph["workgraph_type"].upper() == "FOR":
            should_run = self.check_for_conditions()
            if not should_run:
                self.set_tasks_state(self.ctx._tasks.keys(), "SKIPPED")

    def setup_ctx_workgraph(self, wgdata: t.Dict[str, t.Any]) -> None:
        """setup the workgraph in the context."""

        self.ctx._tasks = wgdata["tasks"]
        self.ctx._links = wgdata["links"]
        self.ctx._connectivity = wgdata["connectivity"]
        self.ctx._ctrl_links = wgdata["ctrl_links"]
        self.ctx._workgraph = wgdata
        self.ctx._error_handlers = wgdata["error_handlers"]

    def read_wgdata_from_base(self) -> t.Dict[str, t.Any]:
        """Read workgraph data from base.extras."""
        from aiida_workgraph.orm.function_data import PickledLocalFunction

        wgdata = self.node.base.extras.get("_workgraph")
        for name, task in wgdata["tasks"].items():
            wgdata["tasks"][name] = deserialize_unsafe(task)
            for _, input in wgdata["tasks"][name]["inputs"].items():
                if input["property"] is None:
                    continue
                prop = input["property"]
                if isinstance(prop["value"], PickledLocalFunction):
                    prop["value"] = prop["value"].value
        wgdata["error_handlers"] = deserialize_unsafe(wgdata["error_handlers"])
        wgdata["context"] = deserialize_unsafe(wgdata["context"])
        return wgdata

    def update_workgraph_from_base(self) -> None:
        """Update the ctx from base.extras."""
        wgdata = self.read_wgdata_from_base()
        for name, task in wgdata["tasks"].items():
            task["results"] = self.ctx._tasks[name].get("results")
        self.setup_ctx_workgraph(wgdata)

    def get_task(self, name: str):
        """Get task from the context."""
        task = Task.from_dict(self.ctx._tasks[name])
        # update task results
        for output in task.outputs:
            output.value = get_nested_dict(
                self.ctx._tasks[name]["results"],
                output.name,
                default=output.value,
            )
        return task

    def update_task(self, task: Task):
        """Update task in the context.
        This is used in error handlers to update the task parameters."""
        tdata = task.to_dict()
        self.ctx._tasks[task.name]["properties"] = tdata["properties"]
        self.ctx._tasks[task.name]["inputs"] = tdata["inputs"]
        self.reset_task(task.name)

    def get_task_state_info(self, name: str, key: str) -> str:
        """Get task state info from ctx."""

        value = self.ctx._tasks[name].get(key, None)
        if key == "process" and value is not None:
            value = deserialize_unsafe(value)
        return value

    def set_task_state_info(self, name: str, key: str, value: any) -> None:
        """Set task state info to ctx and base.extras.
        We task state to the base.extras, so that we can access outside the engine"""

        if key == "process":
            value = serialize(value)
            self.node.base.extras.set(f"_task_{key}_{name}", value)
        else:
            self.node.base.extras.set(f"_task_{key}_{name}", value)
        self.ctx._tasks[name][key] = value

    def init_ctx(self, wgdata: t.Dict[str, t.Any]) -> None:
        """Init the context from the workgraph data."""
        from aiida_workgraph.utils import update_nested_dict

        # set up the context variables
        self.ctx._max_number_awaitables = (
            wgdata["max_number_jobs"] if wgdata["max_number_jobs"] else 1000000
        )
        self.ctx["_count"] = 0
        for key, value in wgdata["context"].items():
            key = key.replace("__", ".")
            update_nested_dict(self.ctx, key, value)
        # set up the workgraph
        self.setup_ctx_workgraph(wgdata)

    def set_task_results(self) -> None:
        for name, task in self.ctx._tasks.items():
            if self.get_task_state_info(name, "action").upper() == "RESET":
                self.reset_task(task["name"])
            self.update_task_state(name)

    def apply_action(self, msg: dict) -> None:

        if msg["catalog"] == "task":
            self.apply_task_actions(msg)
        else:
            self.report(f"Unknow message type {msg}")

    def apply_task_actions(self, msg: dict) -> None:
        """Apply task actions to the workgraph."""
        action = msg["action"]
        tasks = msg["tasks"]
        self.report(f"Action: {action}. {tasks}")
        if action.upper() == "RESET":
            for name in tasks:
                self.reset_task(name)
        elif action.upper() == "PAUSE":
            for name in tasks:
                self.pause_task(name)
        elif action.upper() == "PLAY":
            for name in tasks:
                self.play_task(name)
        elif action.upper() == "SKIP":
            for name in tasks:
                self.skip_task(name)
        elif action.upper() == "KILL":
            for name in tasks:
                self.kill_task(name)

    def reset_task(
        self,
        name: str,
        reset_process: bool = True,
        recursive: bool = True,
        reset_execution_count: bool = True,
    ) -> None:
        """Reset task state and remove it from the executed task.
        If recursive is True, reset its child tasks."""

        self.set_task_state_info(name, "state", "PLANNED")
        if reset_process:
            self.set_task_state_info(name, "process", None)
        self.remove_executed_task(name)
        # self.logger.debug(f"Task {name} action: RESET.")
        # if the task is a while task, reset its child tasks
        if self.ctx._tasks[name]["metadata"]["node_type"].upper() == "WHILE":
            if reset_execution_count:
                self.ctx._tasks[name]["execution_count"] = 0
            for child_task in self.ctx._tasks[name]["children"]:
                self.reset_task(child_task, reset_process=False, recursive=False)
        if recursive:
            # reset its child tasks
            names = self.ctx._connectivity["child_node"][name]
            for name in names:
                self.reset_task(name, recursive=False)

    def pause_task(self, name: str) -> None:
        """Pause task."""
        self.set_task_state_info(name, "action", "PAUSE")
        self.report(f"Task {name} action: PAUSE.")

    def play_task(self, name: str) -> None:
        """Play task."""
        self.set_task_state_info(name, "action", "")
        self.report(f"Task {name} action: PLAY.")

    def skip_task(self, name: str) -> None:
        """Skip task."""
        self.set_task_state_info(name, "state", "SKIPPED")
        self.report(f"Task {name} action: SKIP.")

    def kill_task(self, name: str) -> None:
        """Kill task.
        This is used to kill the awaitable and monitor task.
        """
        if self.get_task_state_info(name, "state") in ["RUNNING"]:
            if self.ctx._tasks[name]["metadata"]["node_type"].upper() in [
                "AWAITABLE",
                "MONITOR",
            ]:
                try:
                    self._temp["awaitables"][name].cancel()
                    self.set_task_state_info(name, "state", "KILLED")
                    self.report(f"Task {name} action: KILLED.")
                except Exception as e:
                    self.logger.error(f"Error in killing task {name}: {e}")

    def continue_workgraph(self) -> None:
        self.report("Continue workgraph.")
        # self.update_workgraph_from_base()
        task_to_run = []
        for name, task in self.ctx._tasks.items():
            # update task state
            if (
                self.get_task_state_info(task["name"], "state")
                in [
                    "CREATED",
                    "RUNNING",
                    "FINISHED",
                    "FAILED",
                    "SKIPPED",
                ]
                or name in self.ctx._executed_tasks
            ):
                continue
            ready, _ = self.is_task_ready_to_run(name)
            if ready:
                task_to_run.append(name)
        #
        self.report("tasks ready to run: {}".format(",".join(task_to_run)))
        self.run_tasks(task_to_run)

    def update_task_state(self, name: str, success=True) -> None:
        """Update task state when the task is finished."""
        task = self.ctx._tasks[name]
        if success:
            node = self.get_task_state_info(name, "process")
            if isinstance(node, orm.ProcessNode):
                # print(f"set task result: {name} process")
                state = node.process_state.value.upper()
                if node.is_finished_ok:
                    self.set_task_state_info(task["name"], "state", state)
                    if task["metadata"]["node_type"].upper() == "WORKGRAPH":
                        # expose the outputs of all the tasks in the workgraph
                        task["results"] = {}
                        outgoing = node.base.links.get_outgoing()
                        for link in outgoing.all():
                            if isinstance(link.node, ProcessNode) and getattr(
                                link.node, "process_state", False
                            ):
                                task["results"][link.link_label] = link.node.outputs
                    else:
                        task["results"] = node.outputs
                        # self.ctx._new_data[name] = task["results"]
                    self.set_task_state_info(task["name"], "state", "FINISHED")
                    self.task_set_context(name)
                    self.report(f"Task: {name} finished.")
                # all other states are considered as failed
                else:
                    task["results"] = node.outputs
                    self.on_task_failed(name)
            elif isinstance(node, orm.Data):
                #
                output_name = [
                    output_name
                    for output_name in list(task["outputs"].keys())
                    if output_name not in ["_wait", "_outputs"]
                ][0]
                task["results"] = {output_name: node}
                self.set_task_state_info(task["name"], "state", "FINISHED")
                self.task_set_context(name)
                self.report(f"Task: {name} finished.")
            else:
                task.setdefault("results", None)
        else:
            self.on_task_failed(name)
        self.update_parent_task_state(name)

    def on_task_failed(self, name: str) -> None:
        """Handle the case where a task has failed."""
        self.set_task_state_info(name, "state", "FAILED")
        self.set_tasks_state(self.ctx._connectivity["child_node"][name], "SKIPPED")
        self.report(f"Task: {name} failed.")
        self.run_error_handlers(name)

    def update_normal_task_state(self, name, results, success=True):
        """Set the results of a normal task.
        A normal task is created by decorating a function with @task().
        """
        from aiida_workgraph.utils import get_sorted_names

        if success:
            task = self.ctx._tasks[name]
            if isinstance(results, tuple):
                if len(task["outputs"]) != len(results):
                    return self.exit_codes.OUTPUS_NOT_MATCH_RESULTS
                output_names = get_sorted_names(task["outputs"])
                for i, output_name in enumerate(output_names):
                    task["results"][output_name] = results[i]
            elif isinstance(results, dict):
                task["results"] = results
            else:
                output_name = [
                    output_name
                    for output_name in list(task["outputs"].keys())
                    if output_name not in ["_wait", "_outputs"]
                ][0]
                task["results"][output_name] = results
            self.task_set_context(name)
            self.set_task_state_info(name, "state", "FINISHED")
            self.report(f"Task: {name} finished.")
        else:
            self.on_task_failed(name)
        self.update_parent_task_state(name)

    def update_parent_task_state(self, name: str) -> None:
        """Update parent task state."""
        parent_task = self.ctx._tasks[name]["parent_task"]
        if parent_task[0]:
            task_type = self.ctx._tasks[parent_task[0]]["metadata"]["node_type"].upper()
            if task_type == "WHILE":
                self.update_while_task_state(parent_task[0])
            elif task_type == "IF":
                self.update_zone_task_state(parent_task[0])
            elif task_type == "ZONE":
                self.update_zone_task_state(parent_task[0])

    def update_while_task_state(self, name: str) -> None:
        """Update while task state."""
        finished, _ = self.are_childen_finished(name)

        if finished:
            self.report(
                f"Wihle Task {name}: this iteration finished. Try to reset for the next iteration."
            )
            # reset the condition tasks
            for link in self.ctx._tasks[name]["inputs"]["conditions"]["links"]:
                self.reset_task(link["from_node"], recursive=False)
            # reset the task and all its children, so that the task can run again
            # do not reset the execution count
            self.reset_task(name, reset_execution_count=False)

    def update_zone_task_state(self, name: str) -> None:
        """Update zone task state."""
        finished, _ = self.are_childen_finished(name)
        if finished:
            self.set_task_state_info(name, "state", "FINISHED")
            self.update_parent_task_state(name)
            self.report(f"Task: {name} finished.")

    def should_run_while_task(self, name: str) -> tuple[bool, t.Any]:
        """Check if the while task should run."""
        # check the conditions of the while task
        not_excess_max_iterations = (
            self.ctx._tasks[name]["execution_count"]
            < self.ctx._tasks[name]["inputs"]["max_iterations"]["property"]["value"]
        )
        conditions = [not_excess_max_iterations]
        _, kwargs, _, _, _ = self.get_inputs(name)
        if isinstance(kwargs["conditions"], list):
            for condition in kwargs["conditions"]:
                value = get_nested_dict(self.ctx, condition)
                conditions.append(value)
        elif isinstance(kwargs["conditions"], dict):
            for _, value in kwargs["conditions"].items():
                conditions.append(value)
        else:
            conditions.append(kwargs["conditions"])
        return False not in conditions

    def should_run_if_task(self, name: str) -> tuple[bool, t.Any]:
        """Check if the IF task should run."""
        _, kwargs, _, _, _ = self.get_inputs(name)
        flag = kwargs["conditions"]
        if kwargs["invert_condition"]:
            return not flag
        return flag

    def are_childen_finished(self, name: str) -> tuple[bool, t.Any]:
        """Check if the child tasks are finished."""
        task = self.ctx._tasks[name]
        finished = True
        for name in task["children"]:
            if self.get_task_state_info(name, "state") not in [
                "FINISHED",
                "SKIPPED",
                "FAILED",
            ]:
                finished = False
                break
        return finished, None

    def run_error_handlers(self, task_name: str) -> None:
        """Run error handler for a task."""

        node = self.get_task_state_info(task_name, "process")
        if not node or not node.exit_status:
            return
        # error_handlers from the task
        for _, data in self.ctx._tasks[task_name]["error_handlers"].items():
            if node.exit_status in data.get("exit_codes", []):
                handler = data["handler"]
                self.run_error_handler(handler, data, task_name)
                return
        # error_handlers from the workgraph
        for _, data in self.ctx._error_handlers.items():
            if node.exit_code.status in data["tasks"].get(task_name, {}).get(
                "exit_codes", []
            ):
                handler = data["handler"]
                metadata = data["tasks"][task_name]
                self.run_error_handler(handler, metadata, task_name)
                return

    def run_error_handler(self, handler: dict, metadata: dict, task_name: str) -> None:
        from inspect import signature
        from aiida_workgraph.utils import get_executor

        handler, _ = get_executor(handler)
        handler_sig = signature(handler)
        metadata.setdefault("retry", 0)
        self.report(f"Run error handler: {handler.__name__}")
        if metadata["retry"] < metadata["max_retries"]:
            task = self.get_task(task_name)
            try:
                if "engine" in handler_sig.parameters:
                    msg = handler(task, engine=self, **metadata.get("kwargs", {}))
                else:
                    msg = handler(task, **metadata.get("kwargs", {}))
                self.update_task(task)
                if msg:
                    self.report(msg)
                metadata["retry"] += 1
            except Exception as e:
                self.report(f"Error in running error handler: {e}")

    def is_workgraph_finished(self) -> bool:
        """Check if the workgraph is finished.
        For `while` workgraph, we need check its conditions"""
        is_finished = True
        failed_tasks = []
        for name, task in self.ctx._tasks.items():
            # self.update_task_state(name)
            if self.get_task_state_info(task["name"], "state") in [
                "RUNNING",
                "CREATED",
                "PLANNED",
                "READY",
            ]:
                is_finished = False
            elif self.get_task_state_info(task["name"], "state") == "FAILED":
                failed_tasks.append(name)
        if is_finished:
            if self.ctx._workgraph["workgraph_type"].upper() == "WHILE":
                should_run = self.check_while_conditions()
                is_finished = not should_run
            if self.ctx._workgraph["workgraph_type"].upper() == "FOR":
                should_run = self.check_for_conditions()
                is_finished = not should_run
        if is_finished and len(failed_tasks) > 0:
            message = f"WorkGraph finished, but tasks: {failed_tasks} failed. Thus all their child tasks are skipped."
            self.report(message)
            result = ExitCode(302, message)
        else:
            result = None
        return is_finished, result

    def check_while_conditions(self) -> bool:
        """Check while conditions.
        Run all condition tasks and check if all the conditions are True.
        """
        self.report("Check while conditions.")
        if self.ctx._execution_count >= self.ctx._max_iteration:
            self.report("Max iteration reached.")
            return False
        condition_tasks = []
        for c in self.ctx._workgraph["conditions"]:
            task_name, socket_name = c.split(".")
            if "task_name" != "context":
                condition_tasks.append(task_name)
        self.run_tasks(condition_tasks, continue_workgraph=False)
        conditions = []
        for c in self.ctx._workgraph["conditions"]:
            task_name, socket_name = c.split(".")
            if task_name == "context":
                conditions.append(self.ctx[socket_name])
            else:
                conditions.append(self.ctx._tasks[task_name]["results"][socket_name])
        should_run = False not in conditions
        if should_run:
            self.reset()
            self.set_tasks_state(condition_tasks, "SKIPPED")
        return should_run

    def check_for_conditions(self) -> bool:
        condition_tasks = [c[0] for c in self.ctx._workgraph["conditions"]]
        self.run_tasks(condition_tasks)
        conditions = [self.ctx._count < len(self.ctx._sequence)] + [
            self.ctx._tasks[c[0]]["results"][c[1]]
            for c in self.ctx._workgraph["conditions"]
        ]
        should_run = False not in conditions
        if should_run:
            self.reset()
            self.set_tasks_state(condition_tasks, "SKIPPED")
            self.ctx["i"] = self.ctx._sequence[self.ctx._count]
        self.ctx._count += 1
        return should_run

    def remove_executed_task(self, name: str) -> None:
        """Remove labels with name from executed tasks."""
        self.ctx._executed_tasks = [
            label for label in self.ctx._executed_tasks if label.split(".")[0] != name
        ]

    def run_tasks(self, names: t.List[str], continue_workgraph: bool = True) -> None:
        """Run tasks.
        Task type includes: Node, Data, CalcFunction, WorkFunction, CalcJob, WorkChain, GraphBuilder,
        WorkGraph, PythonJob, ShellJob, While, If, Zone, FromContext, ToContext, Normal.

        Here we use ToContext to pass the results of the run to the next step.
        This will force the engine to wait for all the submitted processes to
        finish before continuing to the next step.
        """
        from aiida_workgraph.utils import (
            get_executor,
            create_data_node,
            update_nested_dict_with_special_keys,
        )

        for name in names:
            # skip if the max number of awaitables is reached
            task = self.ctx._tasks[name]
            if task["metadata"]["node_type"].upper() in [
                "CALCJOB",
                "WORKCHAIN",
                "GRAPH_BUILDER",
                "WORKGRAPH",
                "PYTHONJOB",
                "SHELLJOB",
            ]:
                if len(self._awaitables) >= self.ctx._max_number_awaitables:
                    print(
                        MAX_NUMBER_AWAITABLES_MSG.format(
                            self.ctx._max_number_awaitables, name
                        )
                    )
                    continue
            # skip if the task is already executed
            if name in self.ctx._executed_tasks:
                continue
            self.ctx._executed_tasks.append(name)
            print("-" * 60)

            self.report(f"Run task: {name}, type: {task['metadata']['node_type']}")
            executor, _ = get_executor(task["executor"])
            # print("executor: ", executor)
            args, kwargs, var_args, var_kwargs, args_dict = self.get_inputs(name)
            for i, key in enumerate(self.ctx._tasks[name]["args"]):
                kwargs[key] = args[i]
            # update the port namespace
            kwargs = update_nested_dict_with_special_keys(kwargs)
            # print("args: ", args)
            # print("kwargs: ", kwargs)
            # print("var_kwargs: ", var_kwargs)
            # kwargs["meta.label"] = name
            # output must be a Data type or a mapping of {string: Data}
            task["results"] = {}
            if task["metadata"]["node_type"].upper() == "NODE":
                results = self.run_executor(executor, [], kwargs, var_args, var_kwargs)
                self.set_task_state_info(name, "process", results)
                self.update_task_state(name)
                if continue_workgraph:
                    self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() == "DATA":
                for key in self.ctx._tasks[name]["args"]:
                    kwargs.pop(key, None)
                results = create_data_node(executor, args, kwargs)
                self.set_task_state_info(name, "process", results)
                self.update_task_state(name)
                self.ctx._new_data[name] = results
                if continue_workgraph:
                    self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in [
                "CALCFUNCTION",
                "WORKFUNCTION",
            ]:
                kwargs.setdefault("metadata", {})
                kwargs["metadata"].update({"call_link_label": name})
                try:
                    # since aiida 2.5.0, we need to use args_dict to pass the args to the run_get_node
                    if var_kwargs is None:
                        results, process = run_get_node(executor, **kwargs)
                    else:
                        results, process = run_get_node(
                            executor, **kwargs, **var_kwargs
                        )
                    process.label = name
                    # print("results: ", results)
                    self.set_task_state_info(name, "process", process)
                    self.update_task_state(name)
                except Exception as e:
                    self.logger.error(f"Error in task {name}: {e}")
                    self.update_task_state(name, success=False)
                # exclude the current tasks from the next run
                if continue_workgraph:
                    self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["CALCJOB", "WORKCHAIN"]:
                # process = run_get_node(executor, *args, **kwargs)
                kwargs.setdefault("metadata", {})
                kwargs["metadata"].update({"call_link_label": name})
                # transfer the args to kwargs
                if self.get_task_state_info(name, "action").upper() == "PAUSE":
                    self.set_task_state_info(name, "action", "")
                    self.report(f"Task {name} is created and paused.")
                    process = create_and_pause_process(
                        self.runner,
                        executor,
                        kwargs,
                        state_msg="Paused through WorkGraph",
                    )
                    self.set_task_state_info(name, "state", "CREATED")
                    process = process.node
                else:
                    process = self.submit(executor, **kwargs)
                    self.set_task_state_info(name, "state", "RUNNING")
                process.label = name
                self.set_task_state_info(name, "process", process)
                self.to_context(**{name: process})
            elif task["metadata"]["node_type"].upper() in ["GRAPH_BUILDER"]:
                wg = self.run_executor(executor, [], kwargs, var_args, var_kwargs)
                wg.name = name
                wg.group_outputs = self.ctx._tasks[name]["metadata"]["group_outputs"]
                wg.parent_uuid = self.node.uuid
                inputs = wg.prepare_inputs(metadata={"call_link_label": name})
                process = self.submit(WorkGraphEngine, inputs=inputs)
                self.set_task_state_info(name, "process", process)
                self.set_task_state_info(name, "state", "RUNNING")
                self.to_context(**{name: process})
            elif task["metadata"]["node_type"].upper() in ["WORKGRAPH"]:
                from .utils import prepare_for_workgraph_task

                inputs, _ = prepare_for_workgraph_task(task, kwargs)
                process = self.submit(WorkGraphEngine, inputs=inputs)
                self.set_task_state_info(name, "process", process)
                self.set_task_state_info(name, "state", "RUNNING")
                self.to_context(**{name: process})
            elif task["metadata"]["node_type"].upper() in ["PYTHONJOB"]:
                from aiida_workgraph.calculations.python import PythonJob
                from .utils import prepare_for_python_task

                inputs = prepare_for_python_task(task, kwargs, var_kwargs)
                # since aiida 2.5.0, we can pass inputs directly to the submit, no need to use **inputs
                if self.get_task_state_info(name, "action").upper() == "PAUSE":
                    self.set_task_state_info(name, "action", "")
                    self.report(f"Task {name} is created and paused.")
                    process = create_and_pause_process(
                        self.runner,
                        PythonJob,
                        inputs,
                        state_msg="Paused through WorkGraph",
                    )
                    self.set_task_state_info(name, "state", "CREATED")
                    process = process.node
                else:
                    process = self.submit(PythonJob, **inputs)
                    self.set_task_state_info(name, "state", "RUNNING")
                process.label = name
                self.set_task_state_info(name, "process", process)
                self.to_context(**{name: process})
            elif task["metadata"]["node_type"].upper() in ["SHELLJOB"]:
                from aiida_shell.calculations.shell import ShellJob
                from .utils import prepare_for_shell_task

                inputs = prepare_for_shell_task(task, kwargs)
                if self.get_task_state_info(name, "action").upper() == "PAUSE":
                    self.set_task_state_info(name, "action", "")
                    self.report(f"Task {name} is created and paused.")
                    process = create_and_pause_process(
                        self.runner,
                        ShellJob,
                        inputs,
                        state_msg="Paused through WorkGraph",
                    )
                    self.set_task_state_info(name, "state", "CREATED")
                    process = process.node
                else:
                    process = self.submit(ShellJob, **inputs)
                    self.set_task_state_info(name, "state", "RUNNING")
                process.label = name
                self.set_task_state_info(name, "process", process)
                self.to_context(**{name: process})
            elif task["metadata"]["node_type"].upper() in ["WHILE"]:
                # TODO refactor this for while, if and zone
                # in case of an empty zone, it will finish immediately
                if self.are_childen_finished(name)[0]:
                    self.update_while_task_state(name)
                else:
                    # check the conditions of the while task
                    should_run = self.should_run_while_task(name)
                    if not should_run:
                        self.set_task_state_info(name, "state", "FINISHED")
                        self.set_tasks_state(
                            self.ctx._tasks[name]["children"], "SKIPPED"
                        )
                        self.update_parent_task_state(name)
                        self.report(
                            f"While Task {name}: Condition not fullilled, task finished. Skip all its children."
                        )
                    else:
                        task["execution_count"] += 1
                        self.set_task_state_info(name, "state", "RUNNING")
                self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["IF"]:
                # in case of an empty zone, it will finish immediately
                if self.are_childen_finished(name)[0]:
                    self.update_zone_task_state(name)
                else:
                    should_run = self.should_run_if_task(name)
                    if should_run:
                        self.set_task_state_info(name, "state", "RUNNING")
                    else:
                        self.set_tasks_state(task["children"], "SKIPPED")
                        self.update_zone_task_state(name)
                self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["ZONE"]:
                # in case of an empty zone, it will finish immediately
                if self.are_childen_finished(name)[0]:
                    self.update_zone_task_state(name)
                else:
                    self.set_task_state_info(name, "state", "RUNNING")
                self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["FROM_CONTEXT"]:
                # get the results from the context
                results = {"result": getattr(self.ctx, kwargs["key"])}
                task["results"] = results
                self.set_task_state_info(name, "state", "FINISHED")
                self.update_parent_task_state(name)
                self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["TO_CONTEXT"]:
                # get the results from the context
                setattr(self.ctx, kwargs["key"], kwargs["value"])
                self.set_task_state_info(name, "state", "FINISHED")
                self.update_parent_task_state(name)
                self.continue_workgraph()
            elif task["metadata"]["node_type"].upper() in ["AWAITABLE"]:
                for key in self.ctx._tasks[name]["args"]:
                    kwargs.pop(key, None)
                awaitable_target = asyncio.ensure_future(
                    self.run_executor(executor, args, kwargs, var_args, var_kwargs),
                    loop=self.loop,
                )
                awaitable = self.construct_awaitable_function(name, awaitable_target)
                self.set_task_state_info(name, "state", "RUNNING")
                self.to_context(**{name: awaitable})
            elif task["metadata"]["node_type"].upper() in ["MONITOR"]:

                for key in self.ctx._tasks[name]["args"]:
                    kwargs.pop(key, None)
                # add function and interval to the args
                args = [
                    executor,
                    kwargs.pop("interval", 1),
                    kwargs.pop("timeout", 3600),
                    *args,
                ]
                awaitable_target = asyncio.ensure_future(
                    self.run_executor(monitor, args, kwargs, var_args, var_kwargs),
                    loop=self.loop,
                )
                awaitable = self.construct_awaitable_function(name, awaitable_target)
                self.set_task_state_info(name, "state", "RUNNING")
                # save the awaitable to the temp, so that we can kill it if needed
                self._temp["awaitables"][name] = awaitable_target
                self.to_context(**{name: awaitable})
            elif task["metadata"]["node_type"].upper() in ["NORMAL"]:
                # Normal task is created by decoratoring a function with @task()
                if "context" in task["kwargs"]:
                    self.ctx.task_name = name
                    kwargs.update({"context": self.ctx})
                for key in self.ctx._tasks[name]["args"]:
                    kwargs.pop(key, None)
                try:
                    results = self.run_executor(
                        executor, args, kwargs, var_args, var_kwargs
                    )
                    self.update_normal_task_state(name, results)
                except Exception as e:
                    self.logger.error(f"Error in task {name}: {e}")
                    self.update_normal_task_state(name, results=None, success=False)
                if continue_workgraph:
                    self.continue_workgraph()
            else:
                # self.report("Unknow task type {}".format(task["metadata"]["node_type"]))
                return self.exit_codes.UNKNOWN_TASK_TYPE

    def construct_awaitable_function(
        self, name: str, awaitable_target: Awaitable
    ) -> None:
        """Construct the awaitable function."""
        awaitable = Awaitable(
            **{
                "pk": name,
                "action": AwaitableAction.ASSIGN,
                "target": "asyncio.tasks.Task",
                "outputs": False,
            }
        )
        awaitable_target.key = name
        awaitable_target.pk = name
        awaitable_target.action = AwaitableAction.ASSIGN
        awaitable_target.add_done_callback(self._on_awaitable_finished)
        return awaitable

    def get_inputs(
        self, name: str
    ) -> t.Tuple[
        t.List[t.Any],
        t.Dict[str, t.Any],
        t.Optional[t.List[t.Any]],
        t.Optional[t.Dict[str, t.Any]],
        t.Dict[str, t.Any],
    ]:
        """Get input based on the links."""

        args = []
        args_dict = {}
        kwargs = {}
        var_args = None
        var_kwargs = None
        task = self.ctx._tasks[name]
        properties = task.get("properties", {})
        inputs = {}
        for name, input in task["inputs"].items():
            # print(f"input: {input['name']}")
            if len(input["links"]) == 0:
                inputs[name] = self.update_context_variable(input["property"]["value"])
            elif len(input["links"]) == 1:
                link = input["links"][0]
                if self.ctx._tasks[link["from_node"]]["results"] is None:
                    inputs[name] = None
                else:
                    # handle the special socket _wait, _outputs
                    if link["from_socket"] == "_wait":
                        continue
                    elif link["from_socket"] == "_outputs":
                        inputs[name] = self.ctx._tasks[link["from_node"]]["results"]
                    else:
                        inputs[name] = get_nested_dict(
                            self.ctx._tasks[link["from_node"]]["results"],
                            link["from_socket"],
                        )
            # handle the case of multiple outputs
            elif len(input["links"]) > 1:
                value = {}
                for link in input["links"]:
                    item_name = f'{link["from_node"]}_{link["from_socket"]}'
                    # handle the special socket _wait, _outputs
                    if link["from_socket"] == "_wait":
                        continue
                    if self.ctx._tasks[link["from_node"]]["results"] is None:
                        value[item_name] = None
                    else:
                        value[item_name] = self.ctx._tasks[link["from_node"]][
                            "results"
                        ][link["from_socket"]]
                inputs[name] = value
        for name in task.get("args", []):
            if name in inputs:
                args.append(inputs[name])
                args_dict[name] = inputs[name]
            else:
                value = self.update_context_variable(properties[name]["value"])
                args.append(value)
                args_dict[name] = value
        for name in task.get("kwargs", []):
            if name in inputs:
                kwargs[name] = inputs[name]
            else:
                value = self.update_context_variable(properties[name]["value"])
                kwargs[name] = value
        if task["var_args"] is not None:
            name = task["var_args"]
            if name in inputs:
                var_args = inputs[name]
            else:
                value = self.update_context_variable(properties[name]["value"])
                var_args = value
        if task["var_kwargs"] is not None:
            name = task["var_kwargs"]
            if name in inputs:
                var_kwargs = inputs[name]
            else:
                value = self.update_context_variable(properties[name]["value"])
                var_kwargs = value
        return args, kwargs, var_args, var_kwargs, args_dict

    def update_context_variable(self, value: t.Any) -> t.Any:
        # replace context variables

        """Get value from context."""
        if isinstance(value, dict):
            for key, sub_value in value.items():
                value[key] = self.update_context_variable(sub_value)
        elif (
            isinstance(value, str)
            and value.strip().startswith("{{")
            and value.strip().endswith("}}")
        ):
            name = value[2:-2].strip()
            return get_nested_dict(self.ctx, name)
        return value

    def task_set_context(self, name: str) -> None:
        """Export task result to context."""
        from aiida_workgraph.utils import update_nested_dict

        items = self.ctx._tasks[name]["context_mapping"]
        for key, value in items.items():
            result = self.ctx._tasks[name]["results"][key]
            update_nested_dict(self.ctx, value, result)

    def is_task_ready_to_run(self, name: str) -> t.Tuple[bool, t.Optional[str]]:
        """Check if the task ready to run.
        For normal task and a zone task, we need to check its input tasks in the connectivity["zone"].
        For task inside a zone, we need to check if the zone (parent task) is ready.
        """
        parent_task = self.ctx._tasks[name]["parent_task"]
        # input_tasks, parent_task, conditions
        parent_states = [True, True]
        # if the task belongs to a parent zone
        if parent_task[0]:
            state = self.get_task_state_info(parent_task[0], "state")
            if state not in ["RUNNING"]:
                parent_states[1] = False
        # check the input tasks of the zone
        # check if the zone input tasks are ready
        for child_task_name in self.ctx._connectivity["zone"][name]["input_tasks"]:
            if self.get_task_state_info(child_task_name, "state") not in [
                "FINISHED",
                "SKIPPED",
                "FAILED",
            ]:
                parent_states[0] = False
                break

        return all(parent_states), parent_states

    def reset(self) -> None:
        self.ctx._execution_count += 1
        self.set_tasks_state(self.ctx._tasks.keys(), "PLANNED")
        self.ctx._executed_tasks = []

    def set_tasks_state(
        self, tasks: t.Union[t.List[str], t.Sequence[str]], value: str
    ) -> None:
        """Set tasks state"""
        for name in tasks:
            self.set_task_state_info(name, "state", value)
            if "children" in self.ctx._tasks[name]:
                self.set_tasks_state(self.ctx._tasks[name]["children"], value)

    def run_executor(
        self,
        executor: t.Callable,
        args: t.List[t.Any],
        kwargs: t.Dict[str, t.Any],
        var_args: t.Optional[t.List[t.Any]],
        var_kwargs: t.Optional[t.Dict[str, t.Any]],
    ) -> t.Any:
        if var_kwargs is None:
            return executor(*args, **kwargs)
        else:
            return executor(*args, **kwargs, **var_kwargs)

    def save_results_to_extras(self, name: str) -> None:
        """Save the results to the base.extras.
        For the outputs of a Normal task, they are not saved to the database like the calcjob or workchain.
        One temporary solution is to save the results to the base.extras. In order to do this, we need to
        serialize the results
        """
        from aiida_workgraph.utils import get_executor

        results = self.ctx._tasks[name]["results"]
        if results is None:
            return
        datas = {}
        for key, value in results.items():
            # find outptus sockets with the name as key
            output = [
                output
                for output in self.ctx._tasks[name]["outputs"]
                if output["name"] == key
            ]
            if len(output) == 0:
                continue
            output = output[0]
            Executor, _ = get_executor(output["serialize"])
            datas[key] = Executor(value)
        self.node.set_extra(f"nodes__results__{name}", datas)

    def message_receive(
        self, _comm: kiwipy.Communicator, msg: t.Dict[str, t.Any]
    ) -> t.Any:
        """
        Coroutine called when the process receives a message from the communicator

        :param _comm: the communicator that sent the message
        :param msg: the message
        :return: the outcome of processing the message, the return value will be sent back as a response to the sender
        """
        self.logger.debug(
            "Process<%s>: received RPC message with communicator '%s': %r",
            self.pid,
            _comm,
            msg,
        )

        intent = msg[process_comms.INTENT_KEY]

        if intent == process_comms.Intent.PLAY:
            return self._schedule_rpc(self.play)
        if intent == process_comms.Intent.PAUSE:
            return self._schedule_rpc(
                self.pause, msg=msg.get(process_comms.MESSAGE_KEY, None)
            )
        if intent == process_comms.Intent.KILL:
            return self._schedule_rpc(
                self.kill, msg=msg.get(process_comms.MESSAGE_KEY, None)
            )
        if intent == process_comms.Intent.STATUS:
            status_info: t.Dict[str, t.Any] = {}
            self.get_status_info(status_info)
            return status_info
        if intent == "custom":
            return self._schedule_rpc(self.apply_action, msg=msg)

        # Didn't match any known intents
        raise RuntimeError("Unknown intent")

    def finalize(self) -> t.Optional[ExitCode]:
        """"""
        # expose outputs of the workgraph
        group_outputs = {}
        for output in self.ctx._workgraph["metadata"]["group_outputs"]:
            names = output["from"].split(".", 1)
            if names[0] == "context":
                if len(names) == 1:
                    raise ValueError("The output name should be context.key")
                update_nested_dict(
                    group_outputs,
                    output["name"],
                    get_nested_dict(self.ctx, names[1]),
                )
            else:
                # expose the whole outputs of the tasks
                if len(names) == 1:
                    update_nested_dict(
                        group_outputs,
                        output["name"],
                        self.ctx._tasks[names[0]]["results"],
                    )
                else:
                    # expose one output of the task
                    # note, the output may not exist
                    if names[1] in self.ctx._tasks[names[0]]["results"]:
                        update_nested_dict(
                            group_outputs,
                            output["name"],
                            self.ctx._tasks[names[0]]["results"][names[1]],
                        )
        self.out_many(group_outputs)
        # output the new data
        if self.ctx._new_data:
            self.out("new_data", self.ctx._new_data)
        self.out("execution_count", orm.Int(self.ctx._execution_count).store())
        self.report("Finalize workgraph.")
        for _, task in self.ctx._tasks.items():
            if self.get_task_state_info(task["name"], "state") == "FAILED":
                return self.exit_codes.TASK_FAILED
