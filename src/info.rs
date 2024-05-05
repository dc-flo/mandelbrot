//! Print information about all the things.
//!
//! Printing info for any of the main types is as simple as
//! `println("{}", &instance);` as `Display` is implemented for each.
//!
//! Printing algorithm is highly janky (due to laziness -- need to complete
//! for each `*InfoResult` type) so lots of stuff isn't formatted correctly
//! (or at all).
//!
//!

extern crate ocl;

use ocl::core::{OclPrm, ProgramInfo};
use ocl::{
    Buffer, Context, Device, Event, EventList, Image, Kernel, Platform, Program, Queue,
    Result as OclResult, Sampler,
};

const PRINT_DETAILED: bool = true;
// Overrides above for device and program:
const PRINT_DETAILED_DEVICE: bool = false;
const PRINT_DETAILED_PROGRAM: bool = false;

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
    __kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

pub(crate) fn info() -> OclResult<()> {
    let dims = 2048;
    let platforms = Platform::list();

    println!("Looping through avaliable platforms ({}):", platforms.len());

    // Loop through all avaliable platforms:
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];

        let devices = Device::list_all(platform)?;

        if devices.is_empty() {
            continue;
        }

        // [NOTE]: A new context can also be created for each device if desired.
        let context = Context::builder()
            .platform(platform.clone())
            .devices(&devices)
            .build()?;

        print_platform_info(&platform)?;

        for device in devices.iter() {
            print_device_info(device)?;
        }

        // Loop through each device
        for d_idx in 0..devices.len() {
            let device = devices[d_idx];

            let queue = Queue::new(&context, device, Some(ocl::core::QUEUE_PROFILING_ENABLE))?;
            let buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .len(dims)
                .build()?;
            let image = Image::<u8>::builder()
                .dims(dims)
                .queue(queue.clone())
                .build()?;
            let sampler = Sampler::with_defaults(&context)?;
            let program = Program::builder()
                .src(SRC)
                .devices(device)
                .build(&context)?;
            let kernel = Kernel::builder()
                .name("multiply")
                .program(&program)
                .queue(queue.clone())
                .global_work_size(dims)
                .arg(&buffer)
                .arg(10.0f32)
                .build()?;

            let mut event_list = EventList::new();
            unsafe {
                kernel.cmd().enew(&mut event_list).enq()?;
            }
            event_list.wait_for()?;

            let mut event = Event::empty();
            buffer
                .cmd()
                .write(&vec![0.0; dims])
                .enew(&mut event)
                .enq()?;
            event.wait_for()?;

            // Print all but device (just once per platform):
            if d_idx == 0 {
                print_context_info(&context);
                print_queue_info(&queue);
                print_buffer_info(&buffer);
                print_image_info(&image);
                print_sampler_info(&sampler);
                print_program_info(&program)?;
                print_kernel_info(&kernel);
                print_event_list_info(&event_list);
                print_event_info(&event);
            }
        }
    }
    Ok(())
}

fn print_platform_info(platform: &Platform) -> OclResult<()> {
    print!("{}", platform);
    let devices = Device::list_all(platform)?;
    print!(" {{ Total Device Count: {} }}", devices.len());
    print!("\n");
    Ok(())
}

fn print_device_info(device: &Device) -> OclResult<()> {
    if PRINT_DETAILED_DEVICE {
        println!("{}", device);
    } else {
        if !PRINT_DETAILED {
            print!("{t}", t = TAB);
        }
        println!("Device (terse) {{ Name: {}, Vendor: {} }}", device.name()?,
            device.vendor()?);
    }
    Ok(())
}

fn print_context_info(context: &Context) {
    println!("{}", context);
}

fn print_queue_info(queue: &Queue) {
    println!("{}", queue);
}

fn print_buffer_info<T: OclPrm>(buffer: &Buffer<T>) {
    println!("{}", buffer);
}

fn print_image_info<S: OclPrm>(image: &Image<S>) {
    println!("{}", image);
}

fn print_sampler_info(sampler: &Sampler) {
    println!("{}", sampler);
}

fn print_program_info(program: &Program) -> OclResult<()> {
    if PRINT_DETAILED_PROGRAM {
        println!("{}", program);
    } else {
        if !PRINT_DETAILED {
            print!("{t}{t}", t = TAB);
        }
        println!("Program (terse) {{ KernelNames: '{}', NumDevices: {}, ReferenceCount: {}, Context: {} }}",
            program.info(ProgramInfo::KernelNames)?,
            program.info(ProgramInfo::NumDevices)?,
            program.info(ProgramInfo::ReferenceCount)?,
            program.info(ProgramInfo::Context)?,
        );
    }
    Ok(())
}

fn print_kernel_info(kernel: &Kernel) {
    println!("{}", kernel);
}

fn print_event_list_info(event_list: &EventList) {
    println!("{:?}", event_list);
}

fn print_event_info(event: &Event) {
    println!("{}", event);
}

pub fn main() {
    match info() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}