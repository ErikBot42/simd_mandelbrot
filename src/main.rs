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

#[inline(never)]
pub unsafe fn mandel_simd(cx: __m256, cy: __m256, i: usize) -> u8 {
    //(__m256, __m256) {
    let i = 8;

    //let (cx0, cx1) = black_box((cx, cx));
    //let (cy0, cy1) = black_box((cy, cy));

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
    macro_rules! mandel_iter {
        ($cx:expr, $cy:expr) => {{
            let mut x = $cx;
            let mut y = $cy;
            for _ in 0..i {
                // x = x * x - y * y + cx
                // y = 2.0 * x * y + cy)
                let xy = mul!(x, y);
                x = add!(fmsub!(x, x, mul!(y, y)), $cx);
                y = fmadd!(xy, splat!(2.0), $cy);
            }
            fmadd!(x, x, mul!(y, y))
        }};
    }

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
    let mask = _mm256_cmp_ps::<1>(fmadd!(x, x, mul!(y, y)), splat!(4.0));
    let b = _mm256_movemask_ps(mask) as u8;
    b
}

fn mandel((x, y): (f32, f32), (cx, cy): (f32, f32)) -> (f32, f32) {
    (x * x - y * y + cx, 2.0 * x * y + cy)
}

fn mandel_i(c: (f32, f32), i: usize) -> bool {
    let mut p = c;
    for a in 0..i {
        p = mandel(p, c);
    }
    return (p.0 * p.0 + p.1 * p.1 < 4.0); // treat NaN as inf
}

fn main() {
    let a = unsafe { _mm256_set_ps(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8) };
    let b = unsafe { _mm256_set_ps(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5) };
    dbg!(a);
    dbg!(b);

    let iterations = 10;

    let a = unsafe { mandel_simd(black_box(a), black_box(b), black_box(iterations)) };
    dbg!(a);

    //println!("Hello, world!");

    let width = 80 / 8;
    let height = 40;

    for y in 0..height {
        for x in 0..width {
            let v = y;

            let uv = (
                x as f32 / width as f32 * 2.0 - 1.0,
                y as f32 / height as f32 * 2.0 - 1.0,
            );
            print!("{:}", if mandel_i(uv, 9) { "X" } else { " " });
        }
        println!();
    }

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
