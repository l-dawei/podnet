from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=0)

print('hello world')
