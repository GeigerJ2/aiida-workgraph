import pytest
from aiida_workgraph import task, WorkGraph
from aiida import orm


@pytest.mark.usefixtures("started_daemon_client")
def test_while(decorated_add, decorated_multiply, decorated_compare):
    # Create a WorkGraph will repeat itself based on the conditions
    wg = WorkGraph("while_workgraph")
    wg.workgraph_type = "WHILE"
    wg.conditions = ["compare1.result"]
    wg.context = {"n": 1}
    wg.max_iteration = 10
    wg.add_task(decorated_compare, name="compare1", x="{{n}}", y=50)
    multiply1 = wg.add_task(
        decorated_multiply, name="multiply1", x="{{ n }}", y=orm.Int(2)
    )
    add1 = wg.add_task(decorated_add, name="add1", y=3)
    add1.set_context({"result": "n"})
    wg.add_link(multiply1.outputs["result"], add1.inputs["x"])
    wg.submit(wait=True, timeout=100)
    assert wg.execution_count == 4
    assert wg.tasks["add1"].outputs["result"].value == 61


def test_while_graph_builder(decorated_add, decorated_multiply, decorated_compare):
    # Create a WorkGraph will repeat itself based on the conditions
    @task.graph_builder(outputs=[{"name": "result", "from": "context.n"}])
    def my_while(n=0, limit=100):
        wg = WorkGraph("while_workgraph")
        wg.workgraph_type = "WHILE"
        wg.conditions = ["compare1.result"]
        wg.context = {"n": n}
        wg.max_iteration = 10
        wg.add_task(decorated_compare, name="compare1", x="{{n}}", y=orm.Int(limit))
        multiply1 = wg.add_task(
            decorated_multiply, name="multiply1", x="{{ n }}", y=orm.Int(2)
        )
        add1 = wg.add_task(decorated_add, name="add1", y=3)
        add1.set_context({"result": "n"})
        wg.add_link(multiply1.outputs["result"], add1.inputs["x"])
        return wg

    # -----------------------------------------
    wg = WorkGraph("while")
    add1 = wg.add_task(decorated_add, name="add1", x=orm.Int(25), y=orm.Int(25))
    my_while1 = wg.add_task(my_while, n=orm.Int(1))
    add2 = wg.add_task(decorated_add, name="add2", y=orm.Int(2))
    wg.add_link(add1.outputs["result"], my_while1.inputs["limit"])
    wg.add_link(my_while1.outputs["result"], add2.inputs["x"])
    wg.submit(wait=True, timeout=100)
    assert add2.outputs["result"].value == 63
    assert my_while1.node.outputs.execution_count == 4
    assert my_while1.outputs["result"].value == 61


def test_while_max_iteration(decorated_add, decorated_multiply, decorated_compare):
    # Create a WorkGraph will repeat itself based on the conditions
    @task.graph_builder(outputs=[{"name": "result", "from": "context.n"}])
    def my_while(n=0, limit=100):
        wg = WorkGraph("while_workgraph")
        wg.workgraph_type = "WHILE"
        wg.max_iteration = 3
        wg.conditions = ["compare1.result"]
        wg.context = {"n": n}
        wg.add_task(decorated_compare, name="compare1", x="{{n}}", y=orm.Int(limit))
        multiply1 = wg.add_task(
            decorated_multiply, name="multiply1", x="{{ n }}", y=orm.Int(2)
        )
        add1 = wg.add_task(decorated_add, name="add1", y=3)
        add1.set_context({"result": "n"})
        wg.add_link(multiply1.outputs["result"], add1.inputs["x"])
        return wg

    # -----------------------------------------
    wg = WorkGraph("while")
    add1 = wg.add_task(decorated_add, name="add1", x=orm.Int(25), y=orm.Int(25))
    my_while1 = wg.add_task(my_while, n=orm.Int(1))
    add2 = wg.add_task(decorated_add, name="add2", y=orm.Int(2))
    wg.add_link(add1.outputs["result"], my_while1.inputs["limit"])
    wg.add_link(my_while1.outputs["result"], add2.inputs["x"])
    wg.submit(wait=True, timeout=100)
    assert add2.outputs["result"].value < 63
    assert my_while1.node.outputs.execution_count == 3
