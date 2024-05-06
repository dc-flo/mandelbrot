use std::env;

mod demo;
mod compute;
mod info;
mod ocl3;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{args:?}");
    let ret = demo::main(args.get(1).unwrap_or(&"100".to_string()).parse::<u32>().unwrap(),
                         args.get(2).unwrap_or(&"1000".to_string()).parse::<i32>().unwrap());
    println!("{ret:?}")
}
