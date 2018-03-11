from __future__ import absolute_import

import mxnet as mx
import numpy as np

# coding: utf-8
"""TensorBoard functions that can be used to log various status during epoch."""
import logging


class LogMetricsCallback(object):
    def __init__(self, logging_dir, prefix=None):
        self.prefix = prefix
        self.itr = 0
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value, self.itr)
        self.itr += 1


class LossMetric(mx.metric.EvalMetric):
    """
    Calculate precision and recall for bounding box detection

    Parameters
    ----------
    threshold : float
    """

    def __init__(self, conf_threshold=0.85, eps=1e-5, allow_extra_outputs=True):
        self.eps = eps
        super(LossMetric, self).__init__('LossMetric', allow_extra_outputs=allow_extra_outputs)
        self.conf_threshold = conf_threshold

    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = self.eps
        self.sum_tp = self.eps
        self.sum_tn = self.eps
        self.sum_fn = self.eps
        self.sum_fp = self.eps
        self.sum_conf = self.eps
        self.sum_x = self.eps
        self.sum_y = self.eps
        self.sum_h = self.eps
        self.sum_w = self.eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

            Parameters
            ----------
            labels : list of `NDArray`
                The labels of the data.
            preds : list of `NDArray`
                Predicted values.
            """
        self.num_inst += 1
        self.sum_loss = np.mean(preds[0].asnumpy())

        label = labels[0].asnumpy().reshape((-1, 49, 5))
        pred = ((preds[0] + 1) / 2).asnumpy().reshape((-1, 49, 5))

        c_label = label[:, :, 0]
        c_pred = pred[:, :, 0]
        boxes_pred = c_pred > self.conf_threshold
        self.sum_tp = np.sum(c_label * boxes_pred)
        self.sum_tn = np.sum((1 - c_label) * (1 - boxes_pred))
        self.sum_fn = np.sum(c_label * (1 - boxes_pred))
        self.sum_fp = np.sum(boxes_pred * (1 - c_label))

        num_boxes = np.sum(c_label)
        self.sum_conf = np.sum(np.abs(c_pred - c_label)) / \
                        (49 * label.shape[0])
        self.sum_conf_box = np.sum(np.abs(c_pred - c_label) * c_label) / \
                            (np.sum(c_label))
        self.sum_conf_nbox = np.sum(np.abs(c_pred - c_label) * (1 - c_label)) / \
                             (np.sum(1 - c_label))
        self.sum_x = np.sum((np.abs(pred[:, :, 1] - 0.5 - label[:, :, 1])) * c_label) * 16 / num_boxes
        self.sum_y = np.sum((np.abs(pred[:, :, 2] - 0.5 - label[:, :, 2])) * c_label) * 16 / num_boxes
        self.sum_w = np.sum(np.abs(pred[:, :, 3] - label[:, :, 3]) * c_label) \
                     * 224 / num_boxes
        self.sum_h = np.sum(np.abs(pred[:, :, 4] - label[:, :, 4]) * c_label) \
                     * 224 / num_boxes

    def get(self):
        """Gets the current evaluation result.

      Returns
      -------
      names : list of str
         Name of the metrics.
      values : list of float
         Value of the evaluations.
      """
        names = ['c_accuracy', 'c_precision', 'c_recall', 'c_diff', 'c_box_diff', 'c_nbox_diff', 'x_diff', 'y_diff',
                 'w_diff', 'h_diff', 'loss']

        values = []
        values.append((self.sum_tp + self.sum_tn) / (
            self.sum_tp + self.sum_tn + self.sum_fp + self.sum_fn))
        values.append(self.sum_tp / (self.sum_tp + self.sum_fp + 1e-6))
        values.append(self.sum_tp / (self.sum_tp + self.sum_fn + 1e-6))
        values.extend([sum_val for sum_val in
                       (self.sum_conf, self.sum_conf_box, self.sum_conf_nbox, self.sum_x, self.sum_y, self.sum_w,
                        self.sum_h, self.sum_loss)])

        return names, values