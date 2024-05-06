
// Copyright (c) 2021 Via Technology Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, cl_int, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;

const PROGRAM_SOURCE: &str = r#"
        __kernel void mandelbrot(__global float* X, __global float* Y, __global int* RET, int iter) {
            int id = get_global_id(0);
            float x0 = X[id];
            float y0 = Y[id];
            float x = 0;
            float y = 0;
            float x2 = 0;
            float y2 = 0;
            int it = 0;

            float q = pow((x0 - (1/4)), 2) + pow(y0, 2);
            if (q*(q + (x + (1/4))) < (1/4) * pow(y0, 2)) {
                RET[id] = iter;
                return;
            }

            while (pow(x, 2) + pow(y, 2) <= 4 && it < iter) {
                y = 2*x*y + y0;
                x = x2 - y2 + x0;
                x2 = x*x;
                y2 = y*y;
                it = it + 1;
            }

            RET[id] = it;
        }
    "#;

const KERNEL_NAME: &str = "mandelbrot";

pub(crate) fn main(vec: &mut Vec<i32>, res: u32, max_iter: i32) -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME)?;

    /////////////////////////////////////////////////////////////////////
    // Compute data

    // The input data
    let mut vec_x:Vec<f32> = Vec::new();
    let mut vec_y:Vec<f32> = Vec::new();

    for i in 0..3 * res {
        for j in 0..2 * res {
            vec_x.push((i as f32)/(res as f32) - 2.0);
            vec_y.push((j as f32)/(res as f32) - 1.0);
        }
    }

    let arr_size: usize = vec_x.len();

    // Create OpenCL device buffers
    let mut x = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, arr_size, ptr::null_mut())?
    };
    let mut y = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, arr_size, ptr::null_mut())?
    };
    let z = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_WRITE_ONLY, arr_size, ptr::null_mut())?
    };

    // Blocking write
    let _x_write_event = unsafe { queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &vec_x, &[])? };

    // Non-blocking write, wait for y_write_event
    let y_write_event =
        unsafe { queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &vec_y, &[])? };

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&z)
            .set_arg(&max_iter)
            .set_global_work_size(arr_size)
            .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let read_event =
        unsafe { queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, vec, &events)? };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

    Ok(())
}
