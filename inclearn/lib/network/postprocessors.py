from mindspore import nn
import mindspore as ms

"""
just as its name, here are postprocessor methods

some api from PyTorch to MindSpore, eg forward to construct
"""


class FactorScalar(nn.Cell):
    def __init__(self, initial_value=1., **kwargs):
        super(FactorScalar, self).__init__()
        self.factor = ms.Parameter(ms.Tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def construct(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.construct(other)

    def __rmul__(self, other):
        return self.construct(other)


class InvertedFactorScalar(nn.Cell):
    def __init__(self, initial_value=1., **kwargs):
        super().__init__()
        self._factor = ms.Parameter(ms.Tensor(initial_value))

    @property
    def factor(self):
        return 1 / (self._factor + 1e-7)

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def construct(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.construct(other)

    def __rmul__(self, other):
        return self.construct(other)


class HeatedUpScalar(nn.Cell):
    def __init__(self, first_value, last_value, nb_steps, scope="task", **kwargs):
        super().__init__()

        self.scope = scope
        self.first_value = first_value
        self.step = (max(first_value, last_value) - min(first_value, last_value)) / (nb_steps - 1)

        if first_value > last_value:
            self._factor = -1
        else:
            self._factor = 1

        self._increment = 0

        print("Heated-up factor is {} with {} scope.".format(self.factor, self.scope))

    def on_task_end(self):
        if self.scope == "task":
            self._increment += 1
        print("Heated-up factor is {}.".format(self.factor))

    def on_epoch_end(self):
        if self.scope == "epoch":
            self._increment += 1

    @property
    def factor(self):
        return self.first_value + (self._factor * self._increment * self.step)

    def construct(self, inputs):
        return self.factor * inputs


"""
Unit Test
    Since we only use FactorScalar in PODNet, we only test it.
    5.25 Success
    But, the question is: what dose the 'ms.Parameter' do
"""
# postprocessor_kwargs = {'type': 'learned_scaling', 'initial_value': 3.0}
# x = FactorScalar(**postprocessor_kwargs)
# print(x.factor.asnumpy())
