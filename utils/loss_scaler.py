class DynamicLossScaler:
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of 
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has 
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.  
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of 
    always using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.  If ``scale_window`` consecutive iterations take place without an overflow, the loss scale is readjusted to loss_scale*``scale_factor``. 
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait before increasing the loss scale.
    """

    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if 
            # Pytorch's .sum() creates a one-element tensor of the same type as x 
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)