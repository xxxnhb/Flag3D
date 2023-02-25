from torch.nn.parallel import DistributedDataParallel
from mmengine.model.utils import detect_anomalous_params
from mmengine.model.wrappers.distributed import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class BaseDistributedDataParallel(DistributedDataParallel):
    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data, optim_wrapper):
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            score, losses = self(data, mode='train')
        loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(loss, model=self)
        return losses

    def val_step(self, data):
        return self.module.val_step(data)

    def test_step(self, data):
        return self.module.test_step(data)