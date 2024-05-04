extern crate ocl;

use ocl::builders::BufferBuilder;
use ocl::{MemFlags, ProQue};

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
    Ok(())
}

pub(crate) fn mandelbrot(vec: &mut Vec<f32>, res: u32, iter: i32) -> ocl::Result<()> {
    let src = r#"
        __kernel void mandelbrot(__global float* X, __global float* Y, int iter) {
            int id = get_global_id(0);
            float x0 = X[id];
            float y0 = Y[id];
            float x = 0;
            float y = 0;
            for (int i = 0; i < iter; i++) {
                if (pow(x, 2) + pow(y, 2) > 4) {
                    X[id] = i;
                    return;
                }
                int xtemp = pow(x, 2) - pow(y, 2) + x0;
                y = 2*x*y + y0;
                x = xtemp;
            }
            X[id] = iter;
        }
    "#;


    let pro_que = ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()?;

    let mut vecX:Vec<f32> = Vec::new();
    let mut vecY:Vec<f32> = Vec::new();

    for i in 1..3 * res {
        for j in 1..2 * res {
            vecX.push((i as f32)/(res as f32) - 2.0);
            vecY.push((j as f32)/(res as f32) - 1.0);
        }
    }
    println!("{}", vecX.len());
    let bufferX = BufferBuilder::new()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(vecX.len())
        .copy_host_slice(&vecX)
        .build()?;
    println!("{}", bufferX.len());
    let bufferY = BufferBuilder::new()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(vecY.len())
        .copy_host_slice(&vecY)
        .build()?;


    let kernel = pro_que.kernel_builder("mandelbrot")
        .arg(&bufferX)
        .arg(&bufferY)
        .arg(iter)
        .build()?;

    unsafe { kernel.enq()?; }

    bufferX.read(vec).enq()?;
    Ok(())
}