extern crate ocl;

use std::mem::size_of_val;
use std::time::Instant;
use ocl::{Buffer, ProQue};

pub(crate) fn trivial() -> ocl::Result<()> {
    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()?;

    let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que.kernel_builder("add")
        .arg(&buffer)
        .arg(10.0f32)
        .build()?;

    unsafe { kernel.enq()?; }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);

    unsafe { kernel.enq()?; }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}

pub(crate) unsafe fn mandelbrot(vec: &mut Vec<i32>, res: u32, iter: i32) -> ocl::Result<()> {
    let src = r#"
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

    let pro_que = ProQue::builder()
        .src(src)
        .build()?;

    let mut vec_x:Vec<f32> = Vec::new();
    let mut vec_y:Vec<f32> = Vec::new();

    for i in 0..3 * res {
        for j in 0..2 * res {
            vec_x.push((i as f32)/(res as f32) - 2.0);
            vec_y.push((j as f32)/(res as f32) - 1.0);
        }
    }
    let buffer_x = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(vec_x.len())
        .copy_host_slice(&vec_x)
        .build()?;
    
    let buffer_y = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(vec_y.len())
        .copy_host_slice(&vec_y)
        .build()?;

    let buffer_ret = Buffer::<i32>::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(vec.len())
        .use_host_slice(vec)
        .build()?;

    let mut kernel = pro_que.kernel_builder("mandelbrot")
        .arg(&buffer_x)
        .arg(&buffer_y)
        .arg(&buffer_ret)
        .arg(iter)
        .global_work_size(vec.len())
        .build()?;

    println!("{}", vec_x.len());
    println!("{}", vec.len());

    let timer = Instant::now();
    kernel.enq()?;

    buffer_ret.read(vec).enq()?;
    println!("calq took {}", timer.elapsed().as_nanos());
    Ok(())
}