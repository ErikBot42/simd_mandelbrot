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
use std::mem::transmute;
use std::time::Instant;

macro_rules! fmadd {
    ($a:expr, $b:expr, $c:expr) => {
        _mm256_fmadd_ps($a, $b, $c)
    };
}
macro_rules! fmsub {
    ($a:expr, $b:expr, $c:expr) => {
        _mm256_fmsub_ps($a, $b, $c)
    };
}
macro_rules! mul {
    ($a:expr, $b:expr) => {
        _mm256_mul_ps($a, $b)
    };
}
macro_rules! add {
    ($a:expr, $b:expr) => {
        _mm256_add_ps($a, $b)
    };
}
macro_rules! splat {
    ($a:expr) => {
        _mm256_set1_ps($a)
    };
}
macro_rules! mandel_cmp_expand {
    ($x:expr, $y:expr, $aa:tt, $bb:tt, $cc:tt) => {{
        let $aa = mul!($y, $y);
        let $bb = fmadd!($x, $x, $aa);
        let $cc = _mm256_cmp_ps::<1>($bb, splat!(4.0));
        _mm256_movemask_ps($cc) as u8
    }};
}
macro_rules! mandel_cmp {
    ($x:expr, $y:expr) => {{
        _mm256_movemask_ps(_mm256_cmp_ps::<1>(
            fmadd!($x, $x, mul!($y, $y)),
            splat!(4.0),
        )) as u8
    }};
}
macro_rules! mandel_iter_expand {
    ($x:expr, $y:expr, $cx:expr, $cy:expr) => {
        let aa = mul!($y, $y);
        let xy = mul!($x, $y);
        let bb = fmsub!($x, $x, aa);
        $x = add!(bb, $cx);
        $y = fmadd!(xy, splat!(2.0), $cy);
    };
}
macro_rules! mandel_iter {
    ($x:tt, $y:tt, $cx:tt, $cy:tt) => {
        let xy = mul!($x, $y);
        $x = add!(fmsub!($x, $x, mul!($y, $y)), $cx);
        $y = fmadd!(xy, splat!(2.0), $cy);
    };
}

macro_rules! mandel_iter_o_dual {
    ($x:tt, $ya:tt, $yb:tt, $cx:tt, $cy:tt) => {{
        $yb = mul!($x, $ya);
        $ya = fmsub!($ya, $ya, $cx);
        $x = fmsub!($x, $x, $ya);
        $yb = fmadd!($yb, splat!(2.0), $cy);
    }}; //$yb = mul!($ya, $yb);
        //$ya = fmsub!($ya, $ya, $cx);
        //$x = fmsub!($x, $x, $ya);
        //$yb = fmadd!($yb, splat!(2.0), $cy);};
}

macro_rules! mandel_iter_o {
    ($x:tt, $y:tt, $cx:tt, $cy:tt) => {
        let xy = mul!($x, $y);
        $x = fmsub!($x, $x, fmsub!($y, $y, $cx));
        $y = fmadd!(xy, splat!(2.0), $cy);
        //let xy = mul!($x, $y);
        //$y = fmsub!($y, $y, $cx);
        //$x = fmsub!($x, $x, $y);
        //$y = fmadd!(splat!(2.0), xy, $cy);
    };
}

macro_rules! mandel_iter_expand_o {
    ($x:expr, $y:expr, $cx:expr, $cy:expr, $aa:tt, $xy:tt) => {
        let $aa = fmsub!($y, $y, $cx);
        let $xy = mul!($x, $y);
        $x = fmsub!($x, $x, $aa);
        $y = fmadd!($xy, splat!(2.0), $cy);
    };
}

macro_rules! mandel_simd_expand {
    ($x:tt, $y:tt, $cx:tt, $cy:tt, $aa:tt, $xy:tt) => {{
        let mut $x = $cx;
        let mut $y = $cy;
        for _ in 0..8 {
            mandel_iter_expand_o!($x, $y, $cx, $cy, $aa, $xy);
        }
        mandel_cmp!($x, $y)
    }};
}

macro_rules! unroll_8 {
    ($i:tt, $code:block) => {
        let $i = 0;
        $code;
        let $i = 1;
        $code;
        let $i = 2;
        $code;
        let $i = 3;
        $code;
        let $i = 4;
        $code;
        let $i = 5;
        $code;
        let $i = 6;
        $code;
        let $i = 7;
        $code;
    };
}

macro_rules! unroll_4 {
    ($i:tt, $code:block) => {
        let $i = 0;
        $code;
        let $i = 1;
        $code;
        let $i = 2;
        $code;
        let $i = 3;
        $code;
    };
}

macro_rules! unroll_2 {
    ($i:tt, $code:block) => {
        let $i = 0;
        $code;
        let $i = 1;
        $code;
    };
}

#[inline(always)]
pub unsafe fn mandel_simd_2(
    (cx0, cx1): (__m256, __m256),
    (cy0, cy1): (__m256, __m256),
) -> (u8, u8) {
    let mut x0 = cx0;
    let mut y0 = cy0;
    let mut x1 = cx1;
    let mut y1 = cy1;
    for _ in 0..8 {
        mandel_iter!(x0, y0, cx0, cy0);
        mandel_iter!(x1, y1, cx1, cy1);
    }
    (mandel_cmp!(x0, y0), mandel_cmp!(x1, y1))
}

#[inline(always)]
pub unsafe fn mandel_simd_2_a(
    (cx0, cx1): (__m256, __m256),
    (cy0, cy1): (__m256, __m256),
) -> (u8, u8) {
    let mut x0 = cx0;
    let mut y0 = cy0;
    let mut x1 = cx1;
    let mut y1 = cy1;
    for _ in 0..8 {
        //mandel_iter_expand_o!(x0, y0, cx0, cy0, aa0, xy0);
        let aa0 = fmsub!(y0, y0, cx0);
        let aa1 = fmsub!(y1, y1, cx1);
        let xy0 = mul!(x0, y0);
        let xy1 = mul!(x1, y1);
        x0 = fmsub!(x0, x0, aa0);
        x1 = fmsub!(x1, x1, aa1);
        y0 = fmadd!(xy0, splat!(2.0), cy0);
        y1 = fmadd!(xy1, splat!(2.0), cy1);
    }
    let aa0 = mul!(y0, y0);
    let aa1 = mul!(y1, y1);
    let bb0 = fmadd!(x0, x0, aa0);
    let bb1 = fmadd!(x1, x1, aa1);
    let cc0 = _mm256_cmp_ps::<1>(bb0, splat!(4.0));
    let cc1 = _mm256_cmp_ps::<1>(bb1, splat!(4.0));
    (_mm256_movemask_ps(cc0) as u8, _mm256_movemask_ps(cc1) as u8)
}

pub unsafe fn mandel_simd_a(cx: __m256, cy: __m256) -> u8 {
    let mut x = cx;
    let mut y = cy;
    let two = splat!(2.0);
    for _ in 0..8 {
        let aa = mul!(y, y);
        let xy = mul!(x, y);
        let bb = fmsub!(x, x, aa);
        x = add!(bb, cx);
        y = fmadd!(xy, two, cy);
    }
    let four = splat!(2.0);
    let aa = mul!(y, y);
    let bb = fmadd!(x, x, aa);
    let cc = _mm256_cmp_ps::<1>(bb, four);
    _mm256_movemask_ps(cc) as u8
}

#[inline(always)]
pub unsafe fn mandel_simd(cx: __m256, cy: __m256) -> u8 {
    let mut x = cx;
    let mut y = cy;
    for _ in 0..8 {
        // x = x * x - y * y + cx
        // y = 2.0 * x * y + cy)
        //let xy = mul!(x, y);
        //x = add!(fmsub!(x, x, mul!(y, y)), cx);
        //y = fmadd!(xy, splat!(2.0), cy);
        mandel_iter!(x, y, cx, cy);
    }
    // in_set = x * x + y * y < 4.0
    mandel_cmp!(x, y)
}

#[inline(always)]
pub unsafe fn mandel_simd_o(cx: __m256, cy: __m256) -> u8 {
    let mut x = cx;
    let mut y = cy;
    for _ in 0..8 {
        // x = x * x - y * y + cx
        // y = 2.0 * x * y + cy)
        let xy = mul!(x, y);
        x = fmsub!(x, x, fmsub!(y, y, cx));
        y = fmadd!(xy, splat!(2.0), cy);
    }
    // in_set = x * x + y * y < 4.0
    mandel_cmp!(x, y)
}

// x0 "=" 0, x1 "=" 64
#[inline(always)]
unsafe fn mandel_simd_64(x0: f32, x1: f32, y: f32) -> u64 {
    let cy = splat!(y);
    let diff = (x1 - x0) / 64.0;
    let group_diff = (x1 - x0) / 8.0;
    let base_offsets = add!(
        mul!(
            _mm256_set_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            splat!(diff)
        ),
        splat!(x0)
    );

    #[repr(C, align(8))]
    struct A([u8; 8]);

    let mut a = A([0; 8]);

    unroll_4!(ii, {
        //for ii in 0..4 {
        let i0 = ii * 2;
        let i1 = ii * 2 + 1;
        let cx0 = fmadd!(splat!(group_diff), splat!(i0 as f32), base_offsets);
        let cx1 = fmadd!(splat!(group_diff), splat!(i1 as f32), base_offsets);
        let mut x0 = cx0;
        let mut ya0 = cy;
        let mut yb0;
        let mut x1 = cx1;
        let mut ya1 = cy;
        let mut yb1;
        //for _ in 0..4 {
        unroll_4!(_, {

            mandel_iter_o_dual!(x0, ya0, yb0, cx0, cy);
            mandel_iter_o_dual!(x1, ya1, yb1, cx1, cy);

            mandel_iter_o_dual!(x0, yb0, ya0, cx0, cy);
            mandel_iter_o_dual!(x1, yb1, ya1, cx1, cy);

        });
        let s0 = mandel_cmp!(x0, ya0);
        let s1 = mandel_cmp!(x1, ya1);
        *a.0.get_unchecked_mut(8 - i0 - 1) = s0;
        *a.0.get_unchecked_mut(8 - i1 - 1) = s1;
    });

    //for i in 0..8 {
    //    let cx = fmadd!(splat!(group_diff), splat!(i as f32), base_offsets);
    //    let s = mandel_simd_expand!(x, y, cx, cy, aa, xy);
    //    *a.0.get_unchecked_mut(8 - i - 1) = s;
    //}

    transmute(a)
}

unsafe fn main_i() {
    const HEIGHT: usize = 64 * 256;//* 4;
    const WIDTH: usize = 2 * HEIGHT;

    let m_iters = WIDTH * HEIGHT;

    let wid = WIDTH.checked_div(64).unwrap();

    let iters = wid * HEIGHT;
    let mut data = vec![0_u64; iters];

    //let mut index = 0;

    let mut data_ptr = data.as_mut_ptr();

    let t = Instant::now();
    for y in 0..HEIGHT {
        for xx in 0..wid {
            let x = xx;
            {
                let x0 = x as f32 / wid as f32 * 2.0 - 1.0;
                let x1 = x0 + 2.0 as f32 / wid as f32;
                let y = y as f32 / HEIGHT as f32 * 2.0 - 1.0;
                let s = mandel_simd_64(x0, x1, y);
                *data_ptr = s;
                data_ptr = data_ptr.offset(1);
            }
        }
    }
    let t = t.elapsed();

    if false {
        let mut data_ptr = data.as_ptr();
        for _ in 0..HEIGHT {
            for _ in 0..wid {
                print!("{:064b}", *data_ptr);
                data_ptr = data_ptr.offset(1);
            }
            println!();
        }
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

fn main() {
    unsafe {
        main_i();
    }
}
