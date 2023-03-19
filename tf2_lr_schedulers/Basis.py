#!python
import numpy as np
import tensorflow as tf
from functools import partial

def constant_func(learning_rate, step):
    output = tf.ones(step.shape)*learning_rate
    return output

def cosine_annealing_func(step, start, end):
    x = step 
    cosine_annealing = 1 + tf.math.cos(tf.constant(np.pi) * x)
    return end + 0.5 * (start - end) * cosine_annealing

def linear_func(step, start, end):
    x = step
    beta = (end - start)
    linear = beta * x + start
    return linear

class ComposeLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        cycle_size,
        interval_fractions, 
        list_funcs, 
        initial_learning_rate = 0.1,
        scale_mode="cycle",
        final_lr_scale=1.0,
        scale_fn=lambda x: 1.0,
        name="ComposeLR",
    ):  
        """_summary_

        Args:
            cycle_size (_type_): _description_
            interval_fractions (list): # list of shifting points, check if sum(interval_fractions)< =1, otherwise need rescaling. 
            list_funcs (list), # list of functions
            initial_learning_rate (float, optional): _description_. Defaults to 0.1.
            final_lr_scale (float, optional): _description_. Defaults to 1.0.
            scale_fn (_type_, optional): _description_. Defaults to lambdax:1.0.
            name (str, optional): _description_. Defaults to "ComposeLR".
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.interval_fractions = interval_fractions
        self.final_lr_scale = final_lr_scale
        self.name = name
        self._total_steps = cycle_size
        self._interval_steps = [ele*self._total_steps for ele in self.interval_fractions]
        self.list_funcs = list_funcs

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "ComposeLR"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            
            dtype = initial_learning_rate.dtype
            step = tf.cast(step, dtype)
            step = tf.cond( tf.rank(step) == 0,
            lambda: tf.expand_dims(step, axis = 0),
            lambda: step)
            total_steps = tf.cast(self._total_steps, dtype)
            shift_steps = [tf.cast(ele, dtype) for ele in self._interval_steps] 
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)
            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)
            compare = [percentage_complete <= self.interval_fractions[0]]
            interval_cumul = [self.interval_fractions[0]]
            normalized_steps = [step - (cycle -1)*total_steps]
            masks = [tf.cast(compare[0], dtype)]

            for idx in range(1, len(self.interval_fractions)):
                interval_cumul += [interval_cumul[idx-1] + self.interval_fractions[idx]]
                compare += [percentage_complete <= interval_cumul[idx]]
                masks += [tf.cast(compare[idx-1] ^ compare[idx], dtype)]
                normalized_steps += [normalized_steps[idx-1] - tf.squeeze(shift_steps[idx-1])]
            
            masks = tf.stack(masks, axis = 0)
            
            lr_segments = list()
            
            for idx in range(len(self._interval_steps)):
                lr_segments += [self.list_funcs[idx](
                    step = normalized_steps[idx]/ shift_steps[idx]
                )]
            lr_segments = tf.stack(lr_segments, axis = 0)
            lr_res = masks*lr_segments
            lr_res = tf.math.reduce_sum(lr_res, axis = 0)
      
            mode_step = cycle if self.scale_mode == "cycle" else step

            if optimizer == False:
                lr_res = lr_res * self.scale_fn(mode_step)

            return lr_res

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "cycle_size": self.cycle_size,
            "scale_mode": self.scale_mode
        }
    
    
def StepWiseLR(cycle_size, 
               initial_learning_rate = 0.1,
               alpha_factor = 0.2,
               interval_fractions = [0.2, 0.2, 0.2, 0.2, 0.2],
               ):
    
    return ComposeLR(cycle_size = cycle_size,
               initial_learning_rate= initial_learning_rate,
               interval_fractions = interval_fractions,
               list_funcs = list(
                    map(lambda x: partial(
                                        constant_func, 
                    learning_rate = initial_learning_rate * alpha_factor ** tf.cast(x, dtype =tf.float32)
                    ), tf.range(len(interval_fractions)))
               ),
               name = "StepWise_Constant"
               )
    
    
def Goyal_LR(cycle_size, 
               initial_learning_rate = 0.1,
               interval_fractions = [0.05, 0.3, 0.3, 0.3],
               warm_up_func = [partial(linear_func, start = 0, end = 0.1)],
               alpha_factor = 0.1,
               cool_down_func = None,
               ):
    len_intervals_stepwise = len(interval_fractions)
    if warm_up_func:
        len_intervals_stepwise -= 1
    
    if cool_down_func:
        len_intervals_stepwise -= 1
        
    
    return ComposeLR(cycle_size = cycle_size,
               initial_learning_rate= initial_learning_rate,
               interval_fractions = interval_fractions,
               list_funcs = warm_up_func + list(
                    map(lambda x: partial(
                                        constant_func, 
                    learning_rate = initial_learning_rate * alpha_factor ** tf.cast(x, dtype =tf.float32)
                    ), tf.range(len_intervals_stepwise))
               ),
               name = "Goyal_LR"
               )    
    
        
def Cyclical_LR(cycle_size, 
              initial_learning_rate,
              maximum_learning_rate,
              interval_fractions = [0.5, 0.5],
              scale_mode = "cycle",
              scale_fn = lambda x:1.0,
              final_lr_scale = 1.0
              ):
    
    
    return ComposeLR(
        initial_learning_rate = maximum_learning_rate,
        cycle_size = cycle_size,
        interval_fractions= interval_fractions,
        list_funcs = [partial(linear_func, start= initial_learning_rate, 
                              end= maximum_learning_rate), 
                      partial(linear_func, start =maximum_learning_rate, 
                              end=initial_learning_rate)],
        scale_mode= scale_mode,
        scale_fn = scale_fn,
        final_lr_scale= final_lr_scale,
        name= 'Triangluar_Cyclical_LR',
    )

    