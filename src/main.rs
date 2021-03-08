use std::fs;
use std::io::Write;
use derive_more::{Add,Sub, Mul, Div};

// Define struct for all time-evolving quantities
#[derive(Debug, Add, Sub, Mul, Div)]
struct Variables {
    x:  f64,
    y:  f64,
    z:  f64,
    v_x: f64,
    v_y: f64,
    v_z: f64,
}

//---------------------------------------------------------------------------------------------\\
//--------------------------------------- RK4 ALGORITHM ---------------------------------------\\
//---------------------------------------------------------------------------------------------\\

fn rk4(fxx: &dyn Fn(f64) -> f64, 
       fyy: &dyn Fn(f64) -> f64,
       fzz: &dyn Fn(f64) -> f64,
       fvx: &dyn Fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)    -> f64,
       fvy: &dyn Fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)    -> f64,
       fvz: &dyn Fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)    -> f64,
       x: f64, y: f64, z: f64,
       v_x: f64, v_y: f64, v_z: f64,
       dt: f64, g: f64, m_b: f64, a_b: f64,
       v_h: f64, a_h: f64, m_s: f64, a_s: f64,
       b_s: f64, m_g: f64, b_g: f64) -> Variables {

    let kx1 =  fxx(v_x);
    let ky1 =  fyy(v_y);
    let kz1 =  fzz(v_z);
    let kvx1 = fvx(x, y, z, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvy1 = fvy(x, y, z, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvz1 = fvz(x, y, z, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);

    let kx2 =  fxx(v_x + dt * 0.5 * kvx1);
    let ky2 =  fyy(v_y + dt * 0.5 * kvy1);
    let kz2 =  fzz(v_z + dt * 0.5 * kvz1);
    let kvx2 = fvx(x  + dt * 0.5 * kx1, y + dt * 0.5 * ky1, z + dt * 0.5 * kz1, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvy2 = fvy(x  + dt * 0.5 * kx1, y + dt * 0.5 * ky1, z + dt * 0.5 * kz1, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvz2 = fvz(x  + dt * 0.5 * kx1, y + dt * 0.5 * ky1, z + dt * 0.5 * kz1, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);

    let kx3 =  fxx(v_x + dt * 0.5 * kvx2);
    let ky3 =  fyy(v_y + dt * 0.5 * kvy2);
    let kz3 =  fzz(v_z + dt * 0.5 * kvz2);
    let kvx3 = fvx(x  + dt * 0.5 * kx2, y + dt * 0.5 * ky2, z + dt * 0.5 * kz2, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvy3 = fvy(x  + dt * 0.5 * kx2, y + dt * 0.5 * ky2, z + dt * 0.5 * kz2, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvz3 = fvz(x  + dt * 0.5 * kx2, y + dt * 0.5 * ky2, z + dt * 0.5 * kz2, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);

    let kx4 =  fxx(v_x + dt * kvx3);
    let ky4 =  fyy(v_y + dt * kvy3);
    let kz4 =  fzz(v_z + dt * kvz3);
    let kvx4 = fvx(x  + dt * kx3, y + dt * ky3, z + dt * kz3, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvy4 = fvy(x  + dt * kx3, y + dt * ky3, z + dt * kz3, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);
    let kvz4 = fvz(x  + dt * kx3, y + dt * ky3, z + dt * kz3, g, m_b, a_b, m_g, a_s, b_g, m_s, b_s, v_h, a_h);

    return Variables {
        x:  x  + (1.0 / 6.0) * dt * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4),
        y:  y  + (1.0 / 6.0) * dt * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4),
        z:  z  + (1.0 / 6.0) * dt * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4),
        v_x: v_x + (1.0 / 6.0) * dt * (kvx1 + 2.0 * kvx2 + 2.0 * kvx3 + kvx4),
        v_y: v_y + (1.0 / 6.0) * dt * (kvy1 + 2.0 * kvy2 + 2.0 * kvy3 + kvy4),
        v_z: v_z + (1.0 / 6.0) * dt * (kvz1 + 2.0 * kvz2 + 2.0 * kvz3 + kvz4),
    };
}

//---------------------------------------------------------------------------------------------\\
//---------------------------------------- DERIVATIVES ----------------------------------------\\
//---------------------------------------------------------------------------------------------\\

fn f_x(v_x: f64) -> f64 {
    v_x
}

fn f_y(v_y: f64) -> f64 {
    v_y
}

fn f_z(v_z: f64) -> f64 {
    v_z
}

fn f_vx(x: f64, y: f64, z: f64, g: f64, m_b: f64, a_b: f64, m_g: f64, a_s: f64, b_g: f64, m_s: f64, b_s: f64, v_h: f64, a_h: f64) -> f64 {
    -x * (g * m_b / ((x*x + y*y + z*z + a_b*a_b).powf(3.0/2.0)) +
          g * m_g / ((x*x + y*y + (a_s + (z*z + b_g*b_g).powf(1.0/2.0)).powi(2)).powf(3.0/2.0)) +
          g * m_s / ((x*x + y*y + (a_s + (z*z + b_s*b_s).powf(1.0/2.0)).powi(2)).powf(3.0/2.0)) +
          v_h * v_h / (x*x + y*y + z*z + a_h*a_h))
}

fn f_vy(x: f64, y: f64, z: f64, g: f64, m_b: f64, a_b: f64, m_g: f64, a_s: f64, b_g: f64, m_s: f64, b_s: f64, v_h: f64, a_h: f64) -> f64 {
    -y * (g * m_b / ((x*x + y*y + z*z + a_b*a_b).powf(3.0/2.0)) +
          g * m_g / ((x*x + y*y + (a_s + (z*z + b_g*b_g).powf(1.0/2.0)).powi(2)).powf(3.0/2.0)) +
          g * m_s / ((x*x + y*y + (a_s + (z*z + b_s*b_s).powf(1.0/2.0)).powi(2)).powf(3.0/2.0)) +
          v_h * v_h / (x*x + y*y + z*z + a_h*a_h))
}

fn f_vz(x: f64, y: f64, z: f64, g: f64, m_b: f64, a_b: f64, m_g: f64, a_s: f64, b_g: f64, m_s: f64, b_s: f64, v_h: f64, a_h: f64) -> f64 {
    -z * (g * m_b / ((x*x + y*y + z*z + a_b*a_b).powf(3.0/2.0)) +
          g * m_g / ((x*x + y*y + (a_s + (z*z + b_g*b_g).powf(1.0/2.0)).powi(2)).powf(3.0/2.0))*(a_s + (z*z + b_g*b_g).powf(1.0/2.0))/((z*z + b_g*b_g).powf(1.0/2.0)) +
          g * m_s / ((x*x + y*y + (a_s + (z*z + b_s*b_s).powf(1.0/2.0)).powi(2)).powf(3.0/2.0))*(a_s + (z*z + b_s*b_s).powf(1.0/2.0))/((z*z + b_s*b_s).powf(1.0/2.0)) +
          v_h * v_h / (x*x + y*y + z*z + a_h*a_h))
}

//---------------------------------------------------------------------------------------------\\
//----------------------------------------- MAIN LOOP -----------------------------------------\\
//---------------------------------------------------------------------------------------------\\

fn main() {
    
    // Model parameters                             //(slr = solar masses)
    let g  = 4.49368236 * 10f64.powf(-12.0);        //gravitational constant (kpc*kpc*kpc/Myr/Myr/slr)
    let m_b = 0.12268000 * 10f64.powf(11.0);        //mass of central bulge (slr)
    let a_b = 0.328530;                             //radial scale length of central bulge (kpc)
    let v_h = 0.1801580941;                         //radial velocities at large distances (kpc/Myr)
    let a_h = 33.26089;                             //radial scale length of dark matter halo (kpc)
    let m_s = 0.884517 * 10f64.powf(11.0);          //mass of thin disk (slr)
    let a_s = 4.383000;                             //radial scale length of thick and thin disks (kpc)
    let b_s = 0.307799;                             //vertical scale length of thin disk (kpc)
    let m_g = 0.087718 * 10f64.powf(11.0);          //mass of thick disk (slr)
    let b_g = 0.986541;                             //vertical scale length of thick disk (kpc)

    // Initial positions and velocities
    let mut x  = 8.0;                               //(kpc)
    let mut y  = 0.0;                               //(kpc)
    let mut z  = 0.0;                               //(kpc)
    let mut v_x = -0.04;                            //(kpc/Myr)
    let mut v_y = ( g * m_s / 8.0 ).powf(1.0/2.0);  //(kpc/Myr)
    let mut v_z = 0.11;                             //(kpc/Myr)

    // Loop parameters
    let step   = 1.0;                               //(Myr)
    let mut t = 0.0;                                //(Myr)
    let t_max = 10000.0;                            //(Myr)

    // Vectors for time and positions
    let mut vect: Vec<f64> = Vec::new();
    let mut vecx: Vec<f64> = Vec::new();
    let mut vecy: Vec<f64> = Vec::new();
    let mut vecz: Vec<f64> = Vec::new();
 
    // Run orbit until maximum time is reached
    while t <= t_max {
 
        // Call RK4 algorithm
        let va = rk4(&f_x, &f_y, &f_z, &f_vx, &f_vy, &f_vz, x, y, z, v_x, v_y, v_z, step, g, m_b, a_b, v_h, a_h, m_s, a_s, b_s, m_g, b_g);

        // Pull positions and velocities from RK4 results
        x  = va.x;
        y  = va.y;
        z  = va.z;
        v_x = va.v_x;
        v_y = va.v_y;
        v_z = va.v_z;

        // Store time and positions
        vect.push(t);
        vecx.push(x);
        vecy.push(y);
        vecz.push(z);

        // Update time and counter
        t += step;

    }

    // Write data file
    let file = fs::File::create("solution.dat").unwrap();
    for i in 0..(vect.len()) {

        writeln!(&file, "{:.6} {:.6} {:.6} {:.6}", vect[i as usize], vecx[i as usize], vecy[i as usize], vecz[i as usize]).unwrap();

    }

}