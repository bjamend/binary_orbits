use std::fs;
use std::io::Write;
use derive_more::{Add,Sub, Mul, Div};

// Define struct for all time-evolving quantities
#[derive(Debug, Add, Sub, Mul, Div)]
struct BallisticParticle {
    x:   f64, // (kpc)
    y:   f64, // (kpc)
    z:   f64, // (kpc)
    v_x: f64, // (kpc/Myr)
    v_y: f64, // (kpc/Myr)
    v_z: f64, // (kpc/Myr)
}

// Define struct for galactic potential model
#[derive(Debug, Add, Sub, Mul, Div)]
struct GalacticPotential {
    g:   f64, // gravitational constant (kpc*kpc*kpc/Myr/Myr/slr)
    m_b: f64, // mass of central bulge (slr)
    a_b: f64, // radial scale length of central bulge (kpc)
    v_h: f64, // radial velocities at large distances (kpc/Myr)
    a_h: f64, // radial scale length of dark matter halo (kpc)
    m_s: f64, // mass of thin disk (slr)
    a_s: f64, // radial scale length of thick and thin disks (kpc)
    b_s: f64, // vertical scale length of thin disk (kpc)
    m_g: f64, // mass of thick disk (slr)
    b_g: f64, // vertical scale length of thick disk (kpc)
}

//---------------------------------------------------------------------------------------------\\
//---------------------------------------- DERIVATIVES ----------------------------------------\\
//---------------------------------------------------------------------------------------------\\

impl GalacticPotential {

    fn acceleration(&self, r: (f64, f64, f64)) -> (f64, f64, f64) {
        let Self {g, m_b, a_b, v_h, a_h, m_s, a_s, b_s, m_g, b_g} = self;
        let (x, y, z) = r;

        let ax = -x * (g * m_b / ((x * x + y * y + z * z + a_b * a_b).powf(1.5)) +
                       g * m_g / ((x * x + y * y + (a_s + (z * z + b_g * b_g).sqrt()).powi(2)).powf(1.5)) +
                       g * m_s / ((x * x + y * y + (a_s + (z * z + b_s * b_s).sqrt()).powi(2)).powf(1.5)) +
                       v_h * v_h / (x * x + y * y + z * z + a_h * a_h));

        let ay = -y * (g * m_b / ((x * x + y * y + z * z + a_b * a_b).powf(1.5)) +
                       g * m_g / ((x * x + y * y + (a_s + (z * z + b_g * b_g).sqrt()).powi(2)).powf(1.5)) +
                       g * m_s / ((x * x + y * y + (a_s + (z * z + b_s * b_s).sqrt()).powi(2)).powf(1.5)) +
                       v_h * v_h / (x * x + y * y + z * z + a_h * a_h));
                       
        let az = -z * (g * m_b / ((x * x + y * y + z * z + a_b * a_b).powf(1.5)) +
                       g * m_g / ((x * x + y * y + (a_s + (z * z + b_g * b_g).sqrt()).powi(2)).powf(1.5)) * (a_s + (z * z + b_g * b_g).sqrt()) / ((z * z + b_g * b_g).sqrt()) +
                       g * m_s / ((x * x + y * y + (a_s + (z * z + b_s * b_s).sqrt()).powi(2)).powf(1.5)) * (a_s + (z * z + b_s * b_s).sqrt()) / ((z * z + b_s * b_s).sqrt()) +
                       v_h * v_h / (x * x + y * y + z * z + a_h * a_h));
        (ax, ay, az)
    }

}

fn f_v(vr: (f64, f64, f64)) -> (f64, f64, f64) {
    let (v_x, v_y, v_z) = vr;
    (v_x, v_y, v_z)
}

//---------------------------------------------------------------------------------------------\\
//--------------------------------------- RK4 ALGORITHM ---------------------------------------\\
//---------------------------------------------------------------------------------------------\\

fn rk4(f_v: &dyn Fn((f64, f64, f64)) -> (f64, f64, f64), 
       x: f64, y: f64, z: f64, dt: f64,
       v_x: f64, v_y: f64, v_z: f64, g_p: GalacticPotential) -> BallisticParticle {

    let (kx1, ky1, kz1) =  f_v((v_x, v_y, v_z));
    let (kvx1, kvy1, kvz1) = g_p.acceleration((x, y, z));

    let (kx2, ky2, kz2) = f_v((v_x + 0.5 * dt * kvx1, v_y + 0.5 * dt * kvy1, v_z + 0.5 * dt * kvz1));
    let (kvx2, kvy2, kvz2) = g_p.acceleration((x + 0.5 * dt * kx1, y + 0.5 * dt * ky1, z + 0.5 * dt * kz1));

    let (kx3, ky3, kz3) = f_v((v_x + 0.5 * dt * kvx2, v_y + 0.5 * dt * kvy2, v_z + 0.5 * dt * kvz2));
    let (kvx3, kvy3, kvz3) = g_p.acceleration((x + 0.5 * dt * kx2, y + 0.5 * dt * ky2, z + 0.5 * dt * kz2));

    let (kx4, ky4, kz4) = f_v((v_x + dt * kvx3, v_y + dt * kvy3, v_z + dt * kvz3));
    let (kvx4, kvy4, kvz4) = g_p.acceleration((x + dt * kx3, y + dt * ky3, z + dt * kz3));

    return BallisticParticle {
        x:  x  + (1.0 / 6.0) * dt * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4),
        y:  y  + (1.0 / 6.0) * dt * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4),
        z:  z  + (1.0 / 6.0) * dt * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4),
        v_x: v_x + (1.0 / 6.0) * dt * (kvx1 + 2.0 * kvx2 + 2.0 * kvx3 + kvx4),
        v_y: v_y + (1.0 / 6.0) * dt * (kvy1 + 2.0 * kvy2 + 2.0 * kvy3 + kvy4),
        v_z: v_z + (1.0 / 6.0) * dt * (kvz1 + 2.0 * kvz2 + 2.0 * kvz3 + kvz4),
    };
}

//---------------------------------------------------------------------------------------------\\
//----------------------------------------- MAIN LOOP -----------------------------------------\\
//---------------------------------------------------------------------------------------------\\

fn main() {

    // Vectors for time and positions
    let mut vect: Vec<f64> = Vec::new();
    let mut vecx: Vec<f64> = Vec::new();
    let mut vecy: Vec<f64> = Vec::new();
    let mut vecz: Vec<f64> = Vec::new();

    // Initial positions and velocities
    let mut x   = 8.0;                              //(kpc)
    let mut y   = 0.0;                              //(kpc)
    let mut z   = 0.0;                              //(kpc)
    let mut v_x = -0.04;                            //(kpc/Myr)
    let mut v_y = 0.22;                             //(kpc/Myr)
    let mut v_z = 0.11;                             //(kpc/Myr)

    // Loop parameters
    let step  = 1.0;                                //(Myr)
    let mut t = 0.0;                                //(Myr)
    let t_max = 10000.0;                            //(Myr)
 
    // Run orbit until maximum time is reached
    while t <= t_max {
 
        // Call RK4 algorithm
        let va = rk4(&f_v, x, y, z, step, v_x, v_y, v_z, GalacticPotential{g: 4.49368236e-12,
                     m_b: 0.12268000e+11, a_b: 0.328530, v_h: 0.1801580941, a_h: 33.26089,
                     m_s: 0.884517e11, a_s: 4.383000, b_s: 0.307799, m_g: 0.087718e11, b_g: 0.986541});

        // Pull positions and velocities from RK4 results
        x   = va.x;
        y   = va.y;
        z   = va.z;
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