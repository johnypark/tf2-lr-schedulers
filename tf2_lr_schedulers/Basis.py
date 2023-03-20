#!python
import numpy as np
import tensorflow as tf
from functools import partial

@tf.function(jit_compile=True)
def constant_func(learning_rate, step):
    step_shape = tf.shape(step)
    #print(step_shape)
    output = tf.ones(step_shape)*learning_rate
    return output

@tf.function(jit_compile=True)
def cosine_annealing_func(step, start, end):
    x = step 
    cosine_annealing = 1 + tf.math.cos(tf.constant(np.pi) * x)
    return end + 0.5 * (start - end) * cosine_annealing

@tf.function(jit_compile=True)
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
            interval_steps = [tf.cast(ele, dtype) for ele in self._interval_steps] 
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)
            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)
            compare = [percentage_complete <= self.interval_fractions[0]]
            interval_cumul = [self.interval_fractions[0]]
            normalized_steps = [step - (cycle -1)*total_steps]
            masks = [tf.cast(compare[0], dtype)]
            lr_segments = [self.list_funcs[0](
                step = normalized_steps[0]/ interval_steps[0]
            )]

            for idx in range(1, len(self.interval_fractions)):
                interval_cumul += [interval_cumul[idx-1] + self.interval_fractions[idx]]
                compare += [percentage_complete <= interval_cumul[idx]]
                masks += [tf.cast(compare[idx-1] ^ compare[idx], dtype)]
                normalized_steps += [normalized_steps[idx-1] - tf.squeeze(interval_steps[idx-1])]
                lr_segments += [self.list_funcs[idx](
                    step = normalized_steps[idx]/ interval_steps[idx]
                )]
                
            masks = tf.stack(masks, axis = 0)
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
    

class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(
        self,
        cycle_size,
        list_LR_funcs = [partial(linear_func, 
                    start = 1e-6,
                    end = 1e-2),
                         partial(linear_func, 
                    start = 1e-2, 
                    end = 1e-6)
                    ],
        interval_fractions = [0.3, 0.7], 
        initial_learning_rate = 1e-6,
        maximum_learning_rate = 1e-2,
        scale_mode="cycle",
        final_lr_scale=1.0,
        scale_fn=lambda x: 1.0,
        name="CylicLR",
    ):  
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximum_learning_rate = maximum_learning_rate
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.interval_fractions = interval_fractions
        self.final_lr_scale = final_lr_scale
        self.name = name
        self._total_steps = cycle_size
        self._interval_steps = [ele*self._total_steps for ele in self.interval_fractions]
        self.list_LR_funcs = list_LR_funcs
        
    def xor_matrix(self, num_edge):
        diag_ones = tf.ones(num_edge)
        diag_neg_ones = tf.ones(num_edge-1)*(-1)
        mm = tf.linalg.diag(diag_ones, k = 0) + tf.linalg.diag(diag_neg_ones, k = -1)
        return tf.reshape(mm, (num_edge, num_edge))

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "CyclicLR"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            maximum_learning_rate = tf.cast(self.maximum_learning_rate, dtype)
            step = tf.cast(step, dtype)
            #step = tf.cond( tf.rank(step) == 0,
            #lambda: tf.expand_dims(step, axis = 0),
            #lambda: step)
            total_steps = tf.cast(self._total_steps, dtype)
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)
            normalized_steps = step - (cycle -1)*total_steps
            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)
            
            utm_ones = tf.linalg.band_part(tf.ones(
                                        (len(self.interval_fractions),
                                        len(self.interval_fractions))
                                      ), 0, -1)
            
            interval_cumul = tf.expand_dims(
                                  tf.cast(
                                      self.interval_fractions, 
                                      dtype = utm_ones.dtype),
                                      axis = 0)@utm_ones

            interval_cumul = tf.squeeze(interval_cumul)
            #interval_cumul = tf.cond(tf.reduce_max(interval_cumul) < 1.0,
            #                  lambda: tf.concat([interval_cumul, tf.reshape(tf.constant(1.0),(1,))], axis = -1),
            #                  lambda: interval_cumul
            #                  )
            
            compare = tf.vectorized_map(lambda idx: percentage_complete - tf.gather(interval_cumul, idx), 
                                        tf.range(interval_cumul.shape[0]))
            
            compare = tf.math.sign(compare)
            compare = tf.nn.relu(compare)
            #compare =  tf.map_fn(
            #    lambda idx: percentage_complete < tf.gather(interval_cumul, idx), 
            #        tf.range(interval_cumul.shape[0]),
            #        fn_output_signature=tf.bool
            #        )
            
            tsm = self.xor_matrix(num_edge = tf.shape(compare)[0])
            
            #compare = tf.expand_dims(compare, axis = -1)
            compare = tf.cast(compare, tsm.dtype) #error here: compare = tf.ensure_shape(compare, (2, 9240))

            #ValueError: Shape must be rank 2 but is rank 1 for '{{node AdamW/CylicLR/EnsureShape_1}} = EnsureShape[T=DT_FLOAT, shape=[2,9240]](AdamW/CylicLR/Cast_3)' with input shapes: [?].
            mask = tsm@compare
            #mask = tf.squeeze(mask)
            
            _interval_steps = tf.cast(self._interval_steps, 
                                      dtype = utm_ones.dtype)

            interval_steps_cumul = tf.expand_dims( _interval_steps,
                                      axis = 0)@utm_ones
            interval_steps_cumul = tf.squeeze(interval_steps_cumul) 
            interval_steps_cumul = tf.concat([tf.reshape(tf.constant(0.0),(1,)), interval_steps_cumul], axis = -1)     
            
            tensor_normalized_steps = tf.vectorized_map(#map_fn(
                lambda idx: (
                    normalized_steps - tf.gather(interval_steps_cumul, idx)
                    )/tf.gather(_interval_steps, idx), 
                tf.range(_interval_steps.shape[0])#,
                #fn_output_signature = dtype
            )
                
            lr_seg0 = self.list_LR_funcs[0](step = tf.gather(tensor_normalized_steps,0))
                    #linear_func(
                    #step = tf.gather(tensor_normalized_steps,0),
                    #start = initial_learning_rate,
                    #end = maximum_learning_rate
                    #) 
            
            #print(tf.shape(lr_seg1))
            lr_seg1 = self.list_LR_funcs[1](step = tf.gather(tensor_normalized_steps,1))
                    #linear_func(
                    #step = tf.gather(tensor_normalized_steps,1),
                    #start = maximum_learning_rate, 
                    #end = initial_learning_rate,
                    #) 
            
            #print(tf.shape(lr_seg2))
            
            lr_res = tf.gather(mask,0)*lr_seg0 + tf.gather(mask,1)*lr_seg1
            
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
        
        

class Goyal_style_LR(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(
        self,
        cycle_size,
        interval_fractions = [0.05, 0.3, 0.3, 0.3, 0.05], 
        initial_learning_rate = 1e-6,
        maximum_learning_rate = 1e-2,
        alpha_factor = 1e-1,
        scale_mode="cycle",
        final_lr_scale=1.0,
        scale_fn=lambda x: 1.0,
        name="Goyal",
    ): 
        
        """ Goyal et al.-like learning rate scheduler. 
            SGD LR = 0.1 * (KN)/256 according to Goyal et al. 
            K: number of workers
            N: samples per worker
            B: K*N, batch_size
            Train for 90 epochs and reduce the learnig rate at 30th, 60th, and 80th epoch
            This is the warm-up version of Goyal et al LR. 
            
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximum_learning_rate = maximum_learning_rate
        self.alpha_factor = alpha_factor
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.interval_fractions = interval_fractions
        self.final_lr_scale = final_lr_scale
        self.name = name
        self._total_steps = cycle_size
        self._interval_steps = [ele*self._total_steps for ele in self.interval_fractions]
        
    def xor_matrix(self, num_edge):
        diag_ones = tf.ones(num_edge)
        diag_neg_ones = tf.ones(num_edge-1)*(-1)
        mm = tf.linalg.diag(diag_ones, k = 0) + tf.linalg.diag(diag_neg_ones, k = -1)
        return tf.reshape(mm, (num_edge, num_edge))

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "CyclicLR"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            maximum_learning_rate = tf.cast(self.maximum_learning_rate, dtype)
            step = tf.cast(step, dtype)
            total_steps = tf.cast(self._total_steps, dtype)
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)
            normalized_steps = step - (cycle -1)*total_steps
            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)
            
            utm_ones = tf.linalg.band_part(tf.ones(
                                        (len(self.interval_fractions),
                                        len(self.interval_fractions))
                                      ), 0, -1)
            
            interval_cumul = tf.expand_dims(
                                  tf.cast(
                                      self.interval_fractions, 
                                      dtype = utm_ones.dtype),
                                      axis = 0)@utm_ones

            interval_cumul = tf.squeeze(interval_cumul)
            compare = tf.vectorized_map(lambda idx: percentage_complete < tf.gather(interval_cumul, idx), 
                                        tf.range(interval_cumul.shape[0]))
            
            tsm = self.xor_matrix(num_edge = tf.shape(compare)[0])
            
            compare = tf.expand_dims(compare, axis = -1) # trick to run it on TPU
            compare = tf.cast(compare, tsm.dtype) 
            mask = tsm@compare
            mask = tf.squeeze(mask) #trick to run it on TPU
            
            _interval_steps = tf.cast(self._interval_steps, 
                                      dtype = utm_ones.dtype)

            interval_steps_cumul = tf.expand_dims( _interval_steps,
                                      axis = 0)@utm_ones
            interval_steps_cumul = tf.squeeze(interval_steps_cumul) 
            interval_steps_cumul = tf.concat([tf.reshape(tf.constant(0.0),(1,)), interval_steps_cumul], axis = -1)     
            
            tensor_normalized_steps = tf.vectorized_map(
                lambda idx: (
                    normalized_steps - tf.gather(interval_steps_cumul, idx)
                    )/tf.gather(_interval_steps, idx), 
                tf.range(_interval_steps.shape[0])
            )
                
            warm_up = linear_func(
                    step = tf.gather(tensor_normalized_steps,0),
                    start = initial_learning_rate,
                    end = maximum_learning_rate
                    ) 
            
            peak_lr = constant_func(
                    step = tf.gather(tensor_normalized_steps,1),
                    learning_rate = maximum_learning_rate
                    )
            
            first_decrease = linear_func(
                    step = tf.gather(tensor_normalized_steps,1),
                    learning_rate = maximum_learning_rate * self.alpha_factor
                    ) 
            
            second_decrease = linear_func(
                    step = tf.gather(tensor_normalized_steps,1),
                    start = maximum_learning_rate * self.alpha_factor**2
                    ) 
            
            third_decrease = linear_func(
                    step = tf.gather(tensor_normalized_steps,1),
                    start = maximum_learning_rate *self.alpha_factor**3, 
                    end = initial_learning_rate,
                    ) 
            
            lr_res = tf.gather(mask,0)*warm_up + \
                     tf.gather(mask,1)*peak_lr + \
                     tf.gather(mask, 2)*first_decrease + \
                     tf.gather(mask, 3)*second_decrease + \
                     tf.gather(mask, 4)*third_decrease
            
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
    
    