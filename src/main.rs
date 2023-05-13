#![allow(unused_unsafe)]

// (a + bi) * (c + di)
// ac + adi + bic + bidi
// (ac - bd) + (ad + bc)i

// (a + bi) * (a + bi)
// (aa - bb) + (ab + ba)i
// (aa - bb) + 2abi

//z_(n+1) = (z_n)**2 + c
//
//if length(z) < 1.0
//
//
use core::arch::x86_64::*;
use std::hint::black_box;

use std::time::Instant;

macro_rules! fmadd {
    ($a:expr, $b:expr, $c:expr) => {
        unsafe { _mm256_fmadd_ps($a, $b, $c) }
    };
}
macro_rules! fmsub {
    ($a:expr, $b:expr, $c:expr) => {
        unsafe { _mm256_fmsub_ps($a, $b, $c) }
    };
}
macro_rules! mul {
    ($a:expr, $b:expr) => {
        unsafe { _mm256_mul_ps($a, $b) }
    };
}
macro_rules! add {
    ($a:expr, $b:expr) => {
        unsafe { _mm256_add_ps($a, $b) }
    };
}
macro_rules! splat {
    ($a:expr) => {
        unsafe { _mm256_set1_ps($a) }
    };
}
macro_rules! splat {
    ($a:expr) => {
        unsafe { _mm256_set1_ps($a) }
    };
}
macro_rules! mandel_cmp {
    ($x:expr, $y:expr) => {
        unsafe {
            _mm256_movemask_ps(_mm256_cmp_ps::<1>(
                fmadd!($x, $x, mul!($y, $y)),
                splat!(4.0),
            )) as u8
        }
    };
}

#[inline(always)]
pub unsafe fn mandel_simd(cx: __m256, cy: __m256) -> u8 {
    let i = 8;

    let mut x = cx;
    let mut y = cy;
    for _ in 0..i {
        // x = x * x - y * y + cx
        // y = 2.0 * x * y + cy)
        let xy = mul!(x, y);
        x = add!(fmsub!(x, x, mul!(y, y)), cx);
        y = fmadd!(xy, splat!(2.0), cy);
    }
    // in_set = x * x + y * y < 4.0
    mandel_cmp!(x, y)
}

fn mandel((x, y): (f32, f32), (cx, cy): (f32, f32)) -> (f32, f32) {
    (x * x - y * y + cx, 2.0 * x * y + cy)
}

fn mandel_i(c: (f32, f32), i: usize) -> bool {
    let mut p = c;
    for a in 0..i {
        p = mandel(p, c);
    }
    return p.0 * p.0 + p.1 * p.1 < 4.0; // treat NaN as inf
}

fn main() {
    let fac = 1;

    let width = 128 * fac;
    let height = 64 * fac;

    let wdiff = 1.0 / width as f32 * 2.0;

    let woffsets = unsafe {
        _mm256_set_ps(
            0.0 * wdiff,
            1.0 * wdiff,
            2.0 * wdiff,
            3.0 * wdiff,
            4.0 * wdiff,
            5.0 * wdiff,
            6.0 * wdiff,
            7.0 * wdiff,
        )
    };

    let m_iters = width * height;

    let mut data = vec![0; m_iters / 8]; //Vec::with_capacity(m_iters / 8);

    //let mut index = 0;

    let mut data_ptr = data.as_mut_ptr();

    let t = Instant::now();
    for y in 0..height {
        for x in (0..(width / 8)).map(|x| x * 8) {
            let u = add!(
                add!(splat!(x as f32 / width as f32 * 2.0), woffsets),
                splat!(-1.0)
            );
            let v = splat!(y as f32 / height as f32 * 2.0 - 1.0);

            let s = unsafe { mandel_simd(u, v) };

            //data[index] = s;
            //index += 1;
            unsafe {
                *data_ptr = s;
            }
            unsafe {
                data_ptr = data_ptr.offset(1);
            }
        }
    }
    let t = t.elapsed();

    let mut data = data.iter();
    for _ in 0..height {
        for _ in (0..(width / 8)).map(|x| x * 8) {
            let s = data.next().unwrap();
            print!("{s:08b}");
        }
        println!();
    }

    black_box(data);
    let secs = t.as_secs_f64();
    let nanos = t.as_secs_f64() * 1_000_000_000.0;
    println!("elapsed: {:?} s", secs);
    println!("elapsed/pixel: {:?} ns", nanos / m_iters as f64);
    println!("elapsed/group: {:?} ns", nanos / (m_iters / 8) as f64);

    //let width = 80;
    //let height = 40;

    //for y in 0..height {
    //    for x in 0..width {
    //        let uv = (
    //            x as f32 / width as f32 * 2.0 - 1.0,
    //            y as f32 / height as f32 * 2.0 - 1.0,
    //        );
    //        print!("{:}", if mandel_i(uv, 9) { "X" } else { " " });
    //    }
    //    println!();
    //}
}
