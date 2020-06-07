mod utils;

use rand::distributions::Uniform;
use rand::prelude::*;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, pso!");
}

#[wasm_bindgen]
pub fn rastringin(x: &[f64]) -> f64 {
    use std::f64::consts::PI;
    let a = 10.0;

    a * x.len() as f64
        + x.iter()
            .map(|xi| xi.powi(2) - a * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

#[wasm_bindgen]
pub struct PSO {
    /** position of particles */
    x: Vec<f64>,
    /** velocity of particles */
    v: Vec<f64>,
    /** best known postion of particles */
    p: Vec<f64>,
    /** bst know position of the entire swarm */
    g: Vec<f64>,
    /** Inertial parameter [0, 1] */
    w: f64,
    /** ratio of ps [0, 1] */
    cp: f64,
    /** ratio of g [0, 1]*/
    cg: f64,
    /** number of particles */
    size: usize,
    /** number of dimension */
    dim: usize,
}

#[wasm_bindgen]
impl PSO {
    pub fn new(lower: f64, upper: f64, dim: usize, size: usize, w: f64, cp: f64, cg: f64) -> Self {
        let rng = rand::thread_rng();
        let xrange = Uniform::new_inclusive(lower, upper);
        let vrange = Uniform::new_inclusive(-(upper - lower), upper - lower);

        // initialize the particle's position
        let x = rng
            .sample_iter(&xrange)
            .take(dim * size)
            .collect::<Vec<f64>>();

        // initialize the particle's velocity
        let v = rng
            .sample_iter(&vrange)
            .take(dim * size)
            .collect::<Vec<f64>>();

        // initialize the particle's best known position
        let p = x.clone();

        // update the swarm's best known  position
        let mut g = x[0..dim].to_vec();
        for i in 0..size {
            if rastringin(&x[(i * dim)..((i + 1) * dim)]) < rastringin(&g) {
                g = x[(i * dim)..((i + 1) * dim)].to_vec();
            }
        }

        Self {
            x,
            v,
            p,
            g,
            w,
            cp,
            cg,
            size,
            dim,
        }
    }

    pub fn tick(&mut self) {
        let mut rng = rand::thread_rng();

        for i in 0..self.size {
            for d in 0..self.dim {
                let rp: f64 = rng.gen_range(0.0, 1.0);
                let rg: f64 = rng.gen_range(0.0, 1.0);
                let idx = self.dim * i + d;

                // update the particle's velociy
                self.v[idx] = self.w * self.v[idx]
                    + self.cp * rp * (self.p[idx] - self.x[idx])
                    + self.cg * rg * (self.g[d] - self.x[idx]);

                // update the particle's position
                self.x[idx] += self.v[idx];
            }

            let range = (i * self.dim)..((i + 1) * self.dim);
            if rastringin(&self.x[range.clone()]) < rastringin(&self.p[range.clone()]) {
                // update the particle's best known position
                for d in 0..self.dim {
                    let idx = self.dim * i + d;
                    self.p[idx] = self.x[idx];
                }

                if rastringin(&self.p[range.clone()]) < rastringin(&self.g) {
                    // update the swarm's best known position
                    self.g = self.p[range.clone()].to_vec();
                }
            }
        }
    }

    pub fn particles(&self) -> *const f64 {
        self.x.as_ptr()
    }
}



#[cfg(test)]
mod tests {}
