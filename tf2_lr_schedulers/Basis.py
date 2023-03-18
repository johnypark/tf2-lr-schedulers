#!python
import numpy as np
import tensorflow as tf


class WarmUp:
    
    def __init__(self,
                 init_LR,
                 max_LR,
                 step_size,
                 scale_mode = 'linear',
                 scale_fn = None,
                 data_type = tf.float32
                 ):
        self.iLR = init_LR
        self.mLR = max_LR
        self.step_size = step_size
        self.scale_mode = scale_mode
        self.data_type = data_type
        
    def __call__(self, step):
        x = step
        x = x/(self.step_size)
        beta = (self.mLR - self.iLR)
        fx = beta*x
        return tf.cast(fx, dtype = self.data_type)
        
    def get_config(self):
        return {
        "initial_learning_rate": self.iLR,
        "step_size": self.step_size}
        
    def total_steps(self):
        return self.step_size
            
class StepDecrease:
    
    def __init__(self,
                 max_LR,
                 step_size,
                 change_at, 
                 scale = 0.1,
                 cls_dtype = tf.float32
                 ):
    
        self.mLR = max_LR
        self.step_size = step_size
        assert type(change_at) is list, "argument should be a list."
        for point_value in change_at:
            assert (
                point_value < step_size
                ), "chage_at should be smaller than \
                    the step size ({}=>{})".format(
                        point_value, 
                        step_size)
        self.change_at = change_at
        self.dtype = cls_dtype
        self.scale = scale
        
    def scale_func(self, idx, scale, mask_list):
        mask_mat = tf.cast(mask_list[idx], dtype = tf.int32)
        idx = tf.cast(idx*mask_mat, dtype = self.dtype)
        return scale**idx
       
    def __call__(self, step):
        step  = tf.convert_to_tensor(step)
        if tf.rank(step) == 0:
            step = tf.expand_dims(step, axis = 0)
        step_shape = tf.shape(step)
        compare = list()
        mask = list()
        compare += [step < self.change_at[0]]
        mask += [step < self.change_at[0]]
        
        for idx in range(1, len(self.change_at)):
            compare += [step < self.change_at[idx]]
            mask += [compare[idx]^compare[idx-1]]
        compare = [tf.cast(ten, dtype = self.dtype) for ten in compare]
        mask += [tf.math.add_n(compare)<1]
        mask_range = tf.range(len(self.change_at)+1)
        lr_segments = list(
           map(
            lambda idx: self.scale_func(idx = idx, scale = self.scale, mask_list = mask),
            mask_range)
            )
        output = tf.ones(shape = step_shape)
        for val_segment in lr_segments:
          output *= val_segment
        output *= self.mLR
        return output
    
    def get_config(self):
        return {
        "inital_learning_rate": self.mLR,
        "step_size": self.step_size}
    
    def total_steps(self):
        return self.step_size
    

def apply_funcs2intervals(step, 
                         list_interval,
                         list_funcs,
                         data_type = tf.float32
                         ):
    assert (
        len(list_interval) == len(list_funcs)
    ), "Number of LR functions and intervals must match."
    compare = list()
    mask = list()
    func_output = list()
    curr_thres = list()
    step = tf.cond( tf.rank(step) == 0,
            lambda: tf.expand_dims(step, axis = 0),
            lambda: step)
    
    curr_thres += [list_interval[0]]
    compare += [step < curr_thres[0]]
    mask += [step < curr_thres[0]]
    mask_shape = step.shape
    mask[0].set_shape(mask_shape)
    masked_step = tf.boolean_mask(step, mask[0])
    func_output += [list_funcs[0](masked_step)]
    
    for idx in range(1, len(list_interval)):
        curr_thres += [curr_thres[idx-1] + list_interval[idx]]
        compare += [step < curr_thres[idx]]
        curr_mask = compare[idx] ^ compare[idx - 1]
        curr_mask.set_shape(mask_shape)
        masked_step = tf.boolean_mask(step, curr_mask)
        masked_step = masked_step - curr_thres[idx-1]
        func_output += [list_funcs[idx](masked_step)]
        mask += [curr_mask]
        
    compare = [tf.cast(ele, dtype = data_type) for ele in compare]
    final_mask = tf.math.add_n(compare) < 1
    mask += [final_mask]
    masked_step = tf.boolean_mask(step, final_mask)
    masked_step = masked_step - curr_thres[idx-1]
    func_output += [list_funcs[idx](masked_step)]
    return tf.concat(func_output, axis =0)
  
  
class ConnectLRs:
    
    def __init__(self, 
                 list_of_LRs):
        
        self.step_size = [lr.get_config()["step_size"] for lr in list_of_LRs]
        self.list_of_LRs = list_of_LRs
        print(self.step_size)
        
    def __call__(self, step):
       step  = tf.convert_to_tensor(step)
       output = apply_funcs2intervals(step, 
                             list_interval = self.step_size,
                             list_funcs = self.list_of_LRs,
                             data_type = tf.float32
       )
       return output
        
class Goyal_LR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 steps_per_epoch, 
                 init_LR =0,
                 name = 'Goyal'):
        super(Goyal_LR, self).__init__()
    
        self.initial = WarmUp(init_LR = init_LR,
                max_LR = 0.1,
                step_size = 5*steps_per_epoch)
        self.subseq = StepDecrease(max_LR = 0.1, 
                        step_size = 100*steps_per_epoch,
                        change_at = [30, 60, 90]*steps_per_epoch)
        self.lr_scheduler = ConnectLRs([self.initial, self.subseq])
        self.name = name

    def __call__(self, step, optimizer = False):
      with tf.name_scope(self.name):
        return self.lr_scheduler(step)

    
    
    