extern crate sdl2;

use std::time::Instant;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels;
use sdl2::rect::Point;

use crate::compute::mandelbrot;

const RESOLUTION: u32 = 200;
const SCREEN_WIDTH: u32 = 3 * RESOLUTION;
const SCREEN_HEIGHT: u32 = 2 * RESOLUTION;
const MAX_IT: u32 = 1000;


pub(crate) fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsys = sdl_context.video()?;
    let window = video_subsys
        .window(
            "rust-sdl2_gfx: draw line & FPSManager",
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    canvas.set_draw_color(pixels::Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut lastx = 0;
    let mut lasty = 0;

    let mut events = sdl_context.event_pump()?;

    'main: loop {
        for event in events.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,

                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => {
                    if keycode == Keycode::Escape {
                        break 'main;
                    } else if keycode == Keycode::M {
                        let mut vec = vec![0.0; (SCREEN_HEIGHT*SCREEN_WIDTH) as usize];
                        let ret = mandelbrot(&mut vec, RESOLUTION, 1000);
                        println!("{ret:?}");
                        for i in 1..SCREEN_WIDTH {
                            for j in 1..SCREEN_HEIGHT {
                                canvas.set_draw_color(pixels::Color::RGB((vec.get((i * SCREEN_HEIGHT + j) as usize).unwrap()/1000.0 * 255.0) as u8, 0, 0));
                                let _ = canvas.draw_point(Point::new(i as i32, j as i32));
                            }
                        }
                        canvas.present()
                    } else if keycode == Keycode::Space {
                        let timer = Instant::now();
                        for _i in 1..SCREEN_WIDTH {
                            for _j in 1..SCREEN_HEIGHT {
                                let i = _i as f32;
                                let j = _j as f32;
                                let x0:f32 = i/(RESOLUTION as f32) - 2.0;
                                let y0:f32 = j/(RESOLUTION as f32) - 1.0;
                                let mut x:f32 = 0.0;
                                let mut y:f32 = 0.0;
                                let mut x2:f32 = 0.0;
                                let mut y2:f32 = 0.0;
                                let mut it = 0;

                                //check if in main bulbs
                                let q = (x0 - (1.0/4.0)).powf(2.0) + y0.powf(2.0);
                                if q*(q + (x + (1.0/4.0))) < (1.0/4.0)*y0.powf(2.0) {
                                    canvas.set_draw_color(pixels::Color::RGB(255, 0, 0));
                                    let _ = canvas.draw_point(Point::new(_i as i32, _j as i32));
                                    continue
                                }

                                //escape time algorithm
                                while x.powf(2.0) + y.powf(2.0) <= 4.0 && it < MAX_IT {
                                    y = 2.0*x*y + y0;
                                    x = x2 - y2 + x0;
                                    x2 = x*x;
                                    y2 = y*y;
                                    it = it + 1;
                                }
                                let it = it as f32;
                                canvas.set_draw_color(pixels::Color::RGB((it/1000.0 * 255.0) as u8, 0,0));
                                let _ = canvas.draw_point(Point::new(i as i32, j as i32));
                            }
                        }
                        canvas.present();
                        println!("took {}", timer.elapsed().as_nanos())
                    } else if keycode == Keycode::H {
                        let timer = Instant::now();
                        canvas.set_draw_color(pixels::Color::RGB(255, 0, 0));
                        for _i in 1..SCREEN_WIDTH {
                            for _j in 1..SCREEN_HEIGHT {
                                let x = (_i as f32)/(RESOLUTION as f32)*2.0 - 2.0;
                                let y = (_j as f32)/(RESOLUTION as f32)*2.0 - 2.0;
                                println!("{}: {}", x, y);
                                if x.powf(2.0) + (y - x.powf(2.0 * (1.0/3.0))).powf(2.0) == 1.0 {
                                    let _ = canvas.draw_point(Point::new(_i as i32, _j as i32));
                                }
                            }
                        }
                        canvas.present();
                        println!("took {}", timer.elapsed().as_nanos())
                    }
                }



                Event::MouseButtonDown { x, y, .. } => {
                    let _ = canvas.draw_line(Point::new(lastx, lasty), Point::new(x, y));
                    lastx = x;
                    lasty = y;
                    println!("mouse btn down at ({},{})", x, y);
                    canvas.present();
                }

                _ => {}
            }
        }
    }

    Ok(())
}