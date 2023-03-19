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
        self.iLR = tf.convert_to_tensor(init_LR)
        self.data_type = self.iLR.dtype
        self.mLR = tf.cast(max_LR, self.data_type)
        self.step_size = tf.cast(step_size, self.data_type)
        self.scale_mode = scale_mode
        
    def __call__(self, step):
        x = tf.cast(step, self.data_type)
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
        mask_mat = tf.gather(mask_list, idx)
        mask_mat = tf.cast(mask_mat, dtype = tf.int32)
        idx = tf.cast(idx*mask_mat, dtype = self.dtype)
        return scale**idx
       
    def __call__(self, step):
        step = tf.convert_to_tensor(step)
        step = tf.cond( tf.rank(step) == 0,
            lambda: tf.expand_dims(step, axis = 0),
            lambda: step)
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
        lr_segments = tf.map_fn(
            fn = lambda idx: self.scale_func(
                idx = idx, 
                scale = self.scale, 
                mask_list = mask),
            elems = mask_range,
            fn_output_signature= self.dtype)
        
        output = tf.math.reduce_prod(lr_segments, axis = 0)
        output *= self.mLR
        return output
    
    def get_config(self):
        return {
        "inital_learning_rate": self.mLR,
        "step_size": self.step_size}
    
    def total_steps(self):
        return self.step_size


#https://stackoverflow.com/questions/75323974/performance-difference-between-tf-boolean-mask-and-tf-gather-tf-where
# tf.where + tf.gather is much better than the tf.boolean_mask
def apply_funcs2intervals(step, 
                         list_interval,
                         list_funcs,
                         data_type
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
    condition = step < curr_thres[0]
    compare += [condition]
    mask += [condition]
    where = tf.where(condition)
    masked_step = tf.gather(step, where)
    func_output += [list_funcs[0](masked_step)]
    
    for idx in range(1, len(list_interval)):
        curr_thres += [curr_thres[idx-1] + list_interval[idx]]
        compare += [step < curr_thres[idx]]
        curr_mask = tf.where(compare[idx] ^ compare[idx - 1])
        masked_step = tf.gather(step, curr_mask)
        masked_step = masked_step - curr_thres[idx-1]
        func_output += [list_funcs[idx](masked_step)]
        mask += [curr_mask]
    compare = [tf.cast(ele, dtype = data_type) for ele in compare]
    final_mask = tf.math.add_n(compare) < 1
    final_mask = tf.where(final_mask)
    mask += [final_mask]
    masked_step = tf.gather(step, final_mask)
    masked_step = masked_step - curr_thres[idx-1]
    func_output += [list_funcs[idx](masked_step)]
      
    return tf.squeeze(tf.concat(func_output, axis =0))
  
  
class ConnectLRs:
    
    def __init__(self, 
                 list_of_LRs,
                 max_LR,
                 list_of_intervals = None):
        
        if list_of_intervals:
            self.step_size = list_of_intervals
        else:
            self.step_size = [lr.get_config()["step_size"] for lr in list_of_LRs]
        self.list_of_LRs = list_of_LRs
        self.max_LR = max_LR
        
    def __call__(self, step):
        max_LR = tf.convert_to_tensor(self.max_LR)
        output = apply_funcs2intervals(step, 
                             list_interval = self.step_size,
                             list_funcs = self.list_of_LRs,
                             data_type = max_LR.dtype
        )
        return output
   
   
class ConstantLR:
    def __init__(
        self, 
        learning_rate,
        step_size
        ):
        self.learning_rate = learning_rate
        self.step_size = step_size

    def __call__(self, step):
        LR = self.learning_rate
        #step = tf.cast(step, tf.int32)
        #step_shape = tf.shape(step)
        constant = tf.ones(shape = step.shape)*LR
        return constant
    
    def get_config(self):
        return {
        "learning_rate": self.learning_rate,
        "step_size": self.step_size}
               
class Goyal_LR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 steps_per_epoch, 
                 init_LR =0.0,
                 name = 'Goyal'):
        super(Goyal_LR, self).__init__()
        
        max_LR = 0.1
    
        self.initial = WarmUp(init_LR = init_LR,
                max_LR = 0.1,
                step_size = 5*steps_per_epoch)
        
        self.constant1 = ConstantLR(learning_rate = 0.1,
                                    step_size = 30*steps_per_epoch)
        
        self.constant2 = ConstantLR(learning_rate = 0.1**2,
                                    step_size = 30*steps_per_epoch)
        
        
        self.constant3 = ConstantLR(learning_rate = 0.1**3,
                                    step_size = 30*steps_per_epoch)
        
        
        self.constant4 = ConstantLR(learning_rate = 0.1**4,
                                    step_size = 10*steps_per_epoch)
        
        self.lr_scheduler = ConnectLRs([self.initial, 
                                        self.constant1,
                                        self.constant2,
                                        self.constant3,
                                        self.constant4],
                                        max_LR = max_LR)
        self.name = name

    def __call__(self, step, optimizer = False):
      with tf.name_scope(self.name):
        return self.lr_scheduler(step)
    
class Goyal_style_LR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 max_LR, 
                 rate_decrease,
                 steps_per_epoch, 
                 intervals = [5, 30, 30, 30, 10], 
                 init_LR = 0.0,
                 
                 
                 name = 'Goyal_style'):
        super(Goyal_style_LR, self).__init__()
    
        self.initial = WarmUp(init_LR = init_LR,
                max_LR = max_LR,
                step_size = intervals[0]*steps_per_epoch)
        
        self.constant1 = ConstantLR(learning_rate = max_LR,
                                    step_size = intervals[1]*steps_per_epoch)
        
        self.constant2 = ConstantLR(learning_rate = max_LR * rate_decrease,
                                    step_size = intervals[2]*steps_per_epoch)
        
        
        self.constant3 = ConstantLR(learning_rate = max_LR * rate_decrease **2,
                                    step_size = intervals[3]*steps_per_epoch)
        
        
        self.constant4 = ConstantLR(learning_rate = max_LR *rate_decrease **3,
                                    step_size = intervals[4]*steps_per_epoch)
        
        self.lr_scheduler = ConnectLRs([self.initial, 
                                        self.constant1,
                                        self.constant2,
                                        self.constant3,
                                        self.constant4],
                                       max_LR = max_LR)
        self.name = name

    def __call__(self, step, optimizer = False):
      with tf.name_scope(self.name):
        return self.lr_scheduler(step)
    

class StepWiseLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        cycle_size,
        initial_learning_rate = 0.1,
        alpha_factor = 0.2,
        scale_fn=lambda x: 1.0,
        shift_points=[0.3, 0.3, 0.3],
        scale_mode="cycle",
        final_lr_scale=1.0,
        warm_up = None,
        cool_down = None,
        name=None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.alpha_factor = alpha_factor
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.shift_points = shift_points
        self.final_lr_scale = final_lr_scale
        self.name = name
        # Defines the position of the max lr in steps
        self._total_steps = cycle_size
        self._shift_steps = [ele*self._total_steps for ele in self.shift_points]

    #def get_cosine_annealing(self, start, end, step, step_size_part, cycle):
    #    x = step / step_size_part
    #    cosine_annealing = 1 + tf.math.cos(tf.constant(np.pi) * x)
    #    return end + 0.5 * (start - end) * cosine_annealing

    def get_constant(self, learning_rate, step, step_size_part, cycle):
        x = step / step_size_part
        x.shape
        output = tf.ones(x.shape)*learning_rate
        return output

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "OneCycle"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            step = tf.cast(step, dtype)
            total_steps = tf.cast(self._total_steps, dtype)
            print(self._shift_steps)
            shift_steps = [tf.cast(ele, dtype) for ele in self._shift_steps] 
            # Check in % the cycle
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)

            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)  # percent of iterations done
            
            compare = [percentage_complete <= self.shift_points[0]]
            accum_shift_points = [self.shift_points[0]]
            normalized_steps = [step - (cycle -1)*total_steps]
            masks = [tf.cast(compare, dtype)]

            for idx in range(1, len(self.shift_points)):
                accum_shift_points += [accum_shift_points[idx-1] + self.shift_points[idx]]
                compare += [percentage_complete <= accum_shift_points[idx]]
                masks += [tf.cast(compare[idx-1] ^ compare[idx], dtype)]
                normalized_steps += [normalized_steps[idx-1] - tf.squeeze(shift_steps[idx-1])]
            
            compare = [tf.cast(ele, dtype = dtype) for ele in compare]
            final_mask = tf.cast(tf.math.add_n(compare) < 1, dtype)
            mask += [final_mask]
            normalized_steps += [normalized_steps[idx-1] - tf.squeeze(shift_steps[idx-1])]
            final_step = self._total_steps - sum(self._shift_steps)
            

            lr_0= self.get_constant(
                initial_learning_rate,
                normalized_steps[0],
                shift_steps[0],
                cycle
            )
            lr_1 = self.get_constant(
                initial_learning_rate * self.alpha_factor,
                normalized_steps[1],
                shift_steps[1],
                cycle
            )
            lr_2 = self.get_constant(
                initial_learning_rate * self.alpha_factor**2,
                normalized_steps[2],
                shift_steps[2],
                cycle
            )
            
            lr_3 = self.get_constant(
                initial_learning_rate * self.alpha_factor**3,
                normalized_steps[3],
                final_step,
                cycle
            )

            lr_res = masks[0]*lr_0 + masks[1]*lr_1 + masks[2]*lr_2 + masks[3]*lr_3
          
            mode_step = cycle if self.scale_mode == "cycle" else step

            if optimizer == False:
                lr_res = lr_res * self.scale_fn(mode_step)

            return tf.squeeze(lr_res)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "cycle_size": self.cycle_size,
            "scale_mode": self.scale_mode,
            "shift_peak": self.shift_peak,
            "final_lr_scale": self.final_lr_scale
        }


    
    