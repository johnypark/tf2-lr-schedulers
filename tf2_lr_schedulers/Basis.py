#!python
import numpy as np
import tensorflow as tf


class WarmUp:
    
    def __init__(self,
                 init_LR,
                 max_LR,
                 step_size,
                 scale_mode = 'linear',
                 scale_fn = None
                 ):
        self.iLR = init_LR
        self.mLR = max_LR
        self.step_size = step_size
        self.scale_mode = scale_mode
        
    def __call__(self, step):
        x = step
        x = x/(self.step_size)
        beta = (self.mLR - self.iLR)
        fx = beta*x
        return fx
        
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
       
    
    def __call__(self, step):
        step = tf.constant(step)
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
            lambda x: self.scale**tf.cast(
                x*tf.cast(mask[x], dtype = tf.int32),
                 dtype = tf.float32)
                , mask_range )
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

        
class ConnectLRs:
    
    def __init__(self, 
                 list_of_LRs):
        
        self.step_size = [lr.get_config()["step_size"] for lr in list_of_LRs]
        self.list_of_LRs = list_of_LRs
        
    def __call__(self, step):
        compare = list()
        mask = list()
        lr_out = list()
        curr_thres = self.step_size[0]
        compare += [step < curr_thres]
        mask += [step < curr_thres]
        #print(mask[0].shape)
        #print(step.shape)
        #print(step[mask[0]])
        #print(self.list_of_LRs[0])
        #print(self.list_of_LRs[0](step[mask[0]]))
        step_use = step[mask[0]]
        lr_out += [self.list_of_LRs[0](step_use)]
        for idx in range(1, len(self.step_size)):
          curr_thres += self.step_size[idx]
          compare += [step < curr_thres]
          mask += [compare[idx]^ compare[idx-1]]
          step_use = step[mask[idx]]
          step_use = step_use - self.step_size[idx-1]
          lr_out += [self.list_of_LRs[idx](step_use)]
        mask += [sum(compare)<1]
        step_use = step[mask[idx+1]]
        step_use = step_use - self.step_size[idx-1]
        lr_out += [self.list_of_LRs[idx](step_use)]
        return np.hstack(lr_out)
        

def Goyal_LR(step, steps_per_epoch, init_LR =0):
    initial = WarmUp(init_LR = init_LR,
                max_LR = 0.1,
                step_size = 5*steps_per_epoch)
    subseq = StepDecrease(max_LR = 0.1, 
                        step_size = 100*steps_per_epoch,
                        change_at = [30, 60, 90]*steps_per_epoch)
    lr_scheduler = ConnectLRs([initial, subseq])
    return lr_scheduler(step)


    
    
    